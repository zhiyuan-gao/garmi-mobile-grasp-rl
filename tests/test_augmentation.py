"""Check perturbation frames and prevent noisy actions from becoming BC labels."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from garmi_rl.augmentation import sample_hand_perturbation, approach_label_mask


class AugmentationTests(unittest.TestCase):
    def test_noise_composes_in_hand_frame(self):
        position=np.array([1.,2.,3.]);rotation=Rotation.from_euler('z',90,degrees=True)
        r=sample_hand_perturbation(position,rotation.as_quat()[[3,0,1,2]],np.random.default_rng(42))
        pose=np.asarray(r['pose'])
        np.testing.assert_allclose(rotation.inv().apply(pose[:3]-position),r['translation_hand_m'])
        actual=rotation.inv()*Rotation.from_quat(pose[[4,5,6,3]])
        np.testing.assert_allclose(actual.as_euler('xyz'),r['rotation_hand_rad'])
        self.assertLessEqual(np.abs(r['translation_hand_m']).max(),.04)
        self.assertLessEqual(np.abs(r['rotation_hand_rad']).max(),.2)

    def test_noisy_actions_and_finish_are_not_supervision(self):
        phases=np.array([0,0,0,0,1,1,2])
        noise=np.array([False,True,True,False,False,False])
        np.testing.assert_array_equal(approach_label_mask(phases,noise),[True,False,False,False,False,False])
        np.testing.assert_array_equal(approach_label_mask(phases),[True,True,True,False,False,False])


if __name__=='__main__':unittest.main()
