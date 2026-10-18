"""Joint increment units, limits, label round trips, and differentiable FK checks."""
import unittest
import torch
from garmi_rl.joint_delta import (decode_joint_delta, encode_joint_target,
    JointPointLoss, JOINT_SCALE)
from garmi_rl.expert_actions import PARK, ARM_TARGET_UPPER


class JointDeltaTest(unittest.TestCase):
    def test_increment_units_and_round_trip(self):
        current = torch.tensor(PARK).repeat(3, 1)
        delta = torch.tensor([[-.04]*7,[0.]*7,[.04]*7], dtype=current.dtype)
        encoded = encode_joint_target(current, current+delta)
        torch.testing.assert_close(encoded, delta/JOINT_SCALE)
        torch.testing.assert_close(decode_joint_delta(current,encoded), current+delta)
        torch.testing.assert_close(decode_joint_delta(current,torch.zeros_like(current)),current)

    def test_action_and_joint_limits(self):
        current = torch.tensor(PARK).reshape(1,7)
        torch.testing.assert_close(decode_joint_delta(current,torch.full_like(current,9.)),current+.05)
        near_limit = torch.tensor(ARM_TARGET_UPPER).reshape(1,7)-.001
        torch.testing.assert_close(decode_joint_delta(near_limit,torch.ones_like(current)),near_limit+.001)

    def test_point_loss_and_gradient(self):
        loss = JointPointLoss('cpu')
        x = torch.zeros(2,64,dtype=torch.float64)
        a = torch.zeros(2,10,dtype=torch.float64,requires_grad=True)
        b = torch.full_like(a,.2)
        self.assertEqual(float(loss(a,a,x)),0.)
        self.assertTrue(torch.autograd.gradcheck(lambda z:loss(z,b,x), (a,), eps=1.e-5,atol=1.e-4))
        q = torch.tensor(PARK,dtype=torch.float64).reshape(1,7)
        import numpy as np
        from scipy.spatial.transform import Rotation
        t = np.eye(4)
        for origin,index in loss.chain:
            t = t @ origin.numpy()
            if index is not None:
                r = np.eye(4);r[:3,:3] = Rotation.from_euler('z',float(q[0,index])).as_matrix();t=t@r
        expected = loss.points.numpy() @ t[:3,:3].T+t[:3,3]
        np.testing.assert_allclose(loss.points_at(q).detach().numpy()[0],expected,atol=1.e-7)


if __name__=='__main__':
    unittest.main()
