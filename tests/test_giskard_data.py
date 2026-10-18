"""Contract checks for stage persistence, sampling independence, and expert version isolation."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from garmi_rl.giskard_io import advance_stages, COMMAND_VERSION
from garmi_rl.expert_learning import load_demonstrations


class TestExpertContract(unittest.TestCase):
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
