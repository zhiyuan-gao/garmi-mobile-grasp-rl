"""Contract checks for stage persistence, sampling independence, and expert version isolation."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from garmi_rl.giskard_io import advance_stages, COMMAND_VERSION, snapshot_queries
from garmi_rl.expert_learning import load_demonstrations


class TestExpertContract(unittest.TestCase):
    def test_snapshot_pairs_link_pose_with_link_velocity(self):
        from types import SimpleNamespace as S
        import torch
        link=torch.zeros(1,13);link[0,0]=10.;link[0,3]=1.;link[0,7]=.1
        mixed=link.clone();mixed[0,7]=.9
        robot=S(data=S(root_link_state_w=link,root_state_w=mixed,
                       joint_pos=torch.zeros(1,7),joint_vel=torch.zeros(1,7)),
                find_joints=lambda _: (list(range(7)),None))
        class Scene(dict):
            env_origins=torch.tensor([[10.,0.,0.]])
        scene=Scene(robot=robot,object=S(data=S(root_state_w=link)))
        env=S(scene=scene,action_manager=S(action=torch.zeros(1,10),
              get_term=lambda _:S(velocity=torch.zeros(1,3))),
              cfg=S(joint_delta_actions=False,fixed_finish={'approach_seconds':30.},
                    actions=S(base=S(velocity_at_link=True))),
              fixed_finish=S(phase_steps=torch.zeros(1),aligned_steps=torch.zeros(1)),
              episode_length_buf=torch.zeros(1),step_dt=.05)
        result=snapshot_queries(env,np.zeros((1,64)),[{'object':'cup','reference':'cup-r004'}],
                                [0],[False],'frame',0)[0]['snapshot']['robot_state']
        self.assertEqual(result[0],0.)
        self.assertAlmostEqual(result[7],.1)

    def test_stage_stays_advanced_after_leaving_pregrasp(self):
        obs=np.zeros((2,64));obs[:,33]=1;obs[:,32]=.08
        stages=advance_stages([False,False],obs)
        self.assertTrue(stages.all())
        obs[:,32]=.01
        self.assertTrue(advance_stages(stages,obs).all())
        self.assertFalse(advance_stages([False,False],obs).any())

    def test_rotated_approach_axis(self):
        obs=np.zeros((1,64));obs[0,33:37]=[2**-.5,0,2**-.5,0];obs[0,30]=.08
        # Correct position but wrong orientation must not advance.
        self.assertFalse(advance_stages([False],obs)[0])
        obs[0,33:37]=[1,0,0,0]
        self.assertFalse(advance_stages([False],obs)[0])

    def test_query_subsampling_does_not_restart_stage(self):
        obs=np.zeros((1,64));obs[:,33]=1;stage=np.array([False]);recorded=[]
        for step,distance in enumerate((.3,.08,.05,.01)):
            obs[:,32]=distance
            stage=advance_stages(stage,obs)
            if step in (0,3):recorded.append(bool(stage[0]))
        self.assertEqual(recorded,[False,True])

    def test_unknown_or_mixed_labels_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'index.json'
            path.write_text(json.dumps(dict(dagger_label_sets=[dict(command_version='restarting_cubic')])) )
            with self.assertRaises(ValueError):load_demonstrations(path,'cpu')
            path.write_text(json.dumps(dict(expert_backend='giskard',
                demonstrations=[dict(command_version='template')])))
            with self.assertRaises(AssertionError):load_demonstrations(path,'cpu')


if __name__=='__main__':unittest.main()
