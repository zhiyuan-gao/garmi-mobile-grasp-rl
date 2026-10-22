"""Balanced object assignments, per-episode grasp goals, and supported placements."""
import hashlib
import itertools
import json
from pathlib import Path
import torch
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply


def configure(cfg, settings):
    import isaaclab.sim as sim
    cfg.multi_object = dict(settings)
    manifest = json.loads(Path(settings["manifest"]).read_text())
    selected = {item["name"]: item for item in manifest["objects"]}
    names = settings.get("objects", list(selected))
    cfg.multi_object["objects"] = names
    cfg.scene.replicate_physics = False
    cfg.scene.object.spawn = sim.MultiUsdFileCfg(
        usd_path=[selected[name]["usd"] for name in names], random_choice=False,
        activate_contact_sensors=True,
        rigid_props=sim.RigidBodyPropertiesCfg(solver_position_iteration_count=8,
            solver_velocity_iteration_count=2, max_depenetration_velocity=1.))
    return {**cfg.multi_object, "manifest_sha256": hashlib.sha256(Path(settings["manifest"]).read_bytes()).hexdigest(),
            "assignment": "env index modulo object count; uniform reference sampling per reset"}


class GraspBank:
    def __init__(self, env):
        self.env = env
        self.settings = env.cfg.multi_object
        document = json.loads(Path(self.settings["manifest"]).read_text())
        mapping = {obj["name"]:obj for obj in document["objects"]}
        self.objects = [mapping[name] for name in self.settings["objects"]]
        self.object_ids = torch.arange(env.num_envs, device=env.device) % len(self.objects)
        masses = env.scene["object"].root_physx_view.get_masses().flatten().to(env.device)
        expected = torch.tensor([o["mass"] for o in self.objects],device=env.device)[self.object_ids]
        if not torch.allclose(masses,expected,atol=1.e-5):
            raise ValueError("Spawned object order/mass does not match the reference bank")
        self.reference_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.references = []
        self.tables = []
        self.corners = []
        self.pools = []
        for obj in self.objects:
            for path_key, hash_key in (("mesh","mesh_sha256"),("usd","usd_sha256")):
                if hashlib.sha256(Path(obj[path_key]).read_bytes()).hexdigest() != obj[hash_key]:
                    raise ValueError(f"Asset changed: {obj['name']} {path_key}")
            refs = obj["references"]
            self.references.append(refs)
            self.tables.append(torch.tensor([r["position"]+r["quaternion"] for r in refs], device=env.device))
            self.corners.append(torch.tensor(list(itertools.product(*zip(*obj["bounds"]))), device=env.device))
            split = self.settings.get("split", "train")
            indices = [i for i,r in enumerate(refs) if split == "all" or r.get("split") == split]
            fixed = self.settings.get("reference_ids", {}).get(obj["name"])
            if fixed is not None:
                if not fixed or set(fixed) - {r["id"] for r in refs}:
                    raise ValueError(f"Invalid reference_ids for {obj['name']}: {fixed}")
                indices = [i for i in indices if refs[i]["id"] in fixed]
            if not indices:
                raise ValueError(f"No {split} references for {obj['name']}")
            self.pools.append(torch.tensor(indices, device=env.device, dtype=torch.long))

    def reset(self, ids):
        if hasattr(self.env.cfg, "kitchen"):
            from .kitchen import reset_filtered
            return reset_filtered(self, ids)
        env = self.env
        count = len(ids)
        random = torch.rand(count, 7, device=env.device)
        settings = self.settings
        if settings.get("reset_corners",False):
            vertices=ids//len(self.objects)
            random=torch.stack([((vertices>>j)&1).float() for j in range(7)],dim=1)
        root = env.scene["robot"].data.default_root_state[ids].clone()
        root[:,:3] += env.scene.env_origins[ids]
        ranges = settings.get("base_ranges", [.1,.2,20.])
        root[:,0] += (2*random[:,0]-1)*ranges[0]
        root[:,1] += (2*random[:,1]-1)*ranges[1]
        zeros = torch.zeros(count, device=env.device)
        root[:,3:7] = quat_from_euler_xyz(zeros,zeros,torch.deg2rad((2*random[:,2]-1)*ranges[2]))
        objstate = env.scene["object"].data.default_root_state[ids].clone()
        yaw = torch.deg2rad(-90. + (2*random[:,5]-1)*settings.get("object_yaw_degrees",30.))
        objstate[:,3:7] = quat_from_euler_xyz(zeros,zeros,yaw)
        xy = settings.get("object_xy", [[.79,.99],[-.35,-.15]])
        for objid, obj in enumerate(self.objects):
            mask = self.object_ids[ids] == objid
            targets = ids[mask]
            n = len(targets)
            if not n:
                continue
            pool = self.pools[objid]
            chosen = pool[(random[mask,6]*len(pool)).long().clamp_max(len(pool)-1)]
            self.reference_ids[targets] = chosen
            env.grasp_pos_o[targets] = self.tables[objid][chosen,:3]
            env.grasp_quat_o[targets] = self.tables[objid][chosen,3:]
            corners = self.corners[objid].expand(n,-1,-1)
            rotated = quat_apply(objstate[mask,None,3:7].expand(-1,8,-1).reshape(-1,4),
                                 corners.reshape(-1,3)).reshape(n,8,3)
            lower, upper = rotated.amin(1), rotated.amax(1)
            # Whole object footprint stays inside the table with a 1 cm margin.
            for axis, bounds in enumerate(((.69,1.29),(-.65,.15))):
                lo = torch.maximum(torch.full((n,),xy[axis][0],device=env.device), bounds[0]+.01-lower[:,axis])
                hi = torch.minimum(torch.full((n,),xy[axis][1],device=env.device), bounds[1]-.01-upper[:,axis])
                if torch.any(hi < lo):
                    raise ValueError("Requested placement region cannot support this object")
                objstate[mask,axis] = lo + random[mask,3+axis]*(hi-lo)
            objstate[mask,2] = .945-lower[:,2]+.001
        objstate[:,:3] += env.scene.env_origins[ids]
        objstate[:,7:] = 0
        return root, objstate

    def cases(self):
        env = self.env
        robot = env.scene["robot"].data.root_state_w.clone()
        obj = env.scene["object"].data.root_state_w.clone()
        robot[:,:3] -= env.scene.env_origins
        obj[:,:3] -= env.scene.env_origins
        return [dict(object=self.objects[int(self.object_ids[i])]["name"],
            reference=self.references[int(self.object_ids[i])][int(self.reference_ids[i])]["id"],
            robot_state=robot[i].cpu().tolist(), object_state=obj[i].cpu().tolist())
            for i in range(env.num_envs)]

    def restore(self, cases):
        env = self.env
        if hasattr(env.cfg, "kitchen"):
            from .kitchen import validate_cases
            validate_cases(self, cases)
        for i, case in enumerate(cases):
            objid = int(self.object_ids[i])
            if case["object"] != self.objects[objid]["name"]:
                raise ValueError("Case object does not match spawned geometry")
            index = next(j for j,r in enumerate(self.references[objid]) if r["id"] == case["reference"])
            fixed = self.settings.get("reference_ids", {}).get(case["object"])
            if fixed is not None and case["reference"] not in fixed:
                raise ValueError("Replay reference is outside the frozen task")
            self.reference_ids[i] = index
            env.grasp_pos_o[i] = self.tables[objid][index,:3]
            env.grasp_quat_o[i] = self.tables[objid][index,3:]
        ids = torch.arange(len(cases),device=env.device)
        for key, scene_key in (("robot_state","robot"),("object_state","object")):
            state = torch.tensor([c[key] for c in cases], device=env.device)
            state[:,:3] += env.scene.env_origins[ids]
            env.scene[scene_key].write_root_pose_to_sim(state[:,:7],ids)
            env.scene[scene_key].write_root_velocity_to_sim(state[:,7:],ids)
            if scene_key == "object":
                env.grasp_state["initial_z"][ids] = state[:,2]
        env.sim.forward()
        env.scene.update(0.)
        from . import mdp
        _,_,gp,gq=mdp.geometry(env)
        env.fixed_finish.target_position[:]=gp
        env.fixed_finish.target_rotation[:]=gq
