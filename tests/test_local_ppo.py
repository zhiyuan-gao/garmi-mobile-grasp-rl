import numpy as np
from garmi_rl.local_ppo import strict_pose_gate


def test_strict_ppo_pose_gate_rejects_timeout_close_drift_and_nonfinite():
 trigger=np.array([1,2,1,1,1,1]);distance=np.array([.005,.002,.002,.002,np.nan,.002])
 angle=np.array([.05,.02,.02,.02,.02,.02]);close=np.array([.005,.004,.00501,.004,.004,.004])
 close_angle=np.array([.05,.04,.04,.05001,.04,np.inf])
 assert strict_pose_gate(trigger,distance,angle,close,close_angle).tolist()==[True,False,False,False,False,False]
