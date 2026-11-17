import unittest

import numpy as np

from usv_model import (
    MODEL_JSON_PATH,
    PARAMS,
    PHI_ALLOC,
    THRUSTER_FWD_MAX,
    THRUSTER_REV_MAX,
    THRUSTERS_JSON_PATH,
    cmd_to_thrust_richards,
    thrust_to_normalized_cmd,
    thrust_to_pwm_us,
    thruster_forces_to_pwm_us,
)


class ThrusterPwmAndJsonTests(unittest.TestCase):
    def test_json_files_exist_and_load_9_parameters(self):
        self.assertTrue(MODEL_JSON_PATH.exists())
        self.assertTrue(THRUSTERS_JSON_PATH.exists())
        expected_keys = {"m11", "m22", "m33", "Xu", "Xuu", "Yv", "Yvv", "Nr", "Nrr"}
        self.assertEqual(set(PARAMS.keys()), expected_keys)
        self.assertAlmostEqual(PARAMS["m11"], 219.872606, places=5)
        self.assertAlmostEqual(PARAMS["m22"], 230.539377, places=5)
        self.assertAlmostEqual(PARAMS["m33"], 9.584423, places=5)
        self.assertEqual(PHI_ALLOC.shape, (3, 4))

    def test_reference_thrust_values_map_to_pwm_endpoints(self):
        forces = np.array([-THRUSTER_REV_MAX, 0.0, THRUSTER_FWD_MAX])

        np.testing.assert_allclose(
            thrust_to_pwm_us(forces),
            [1100.0, 1500.0, 1900.0],
            atol=1e-2,
        )

    def test_reference_thrust_values_map_to_normalized_commands(self):
        forces = np.array([-THRUSTER_REV_MAX, 0.0, THRUSTER_FWD_MAX])

        np.testing.assert_allclose(
            thrust_to_normalized_cmd(forces),
            [-1.0, 0.0, 1.0],
            atol=1e-3,
        )

    def test_richards_curve_round_trip(self):
        forces = np.array([-20.0, -10.0, 0.0, 12.0, 25.0])
        cmds = thrust_to_normalized_cmd(forces)
        recovered = cmd_to_thrust_richards(cmds)
        np.testing.assert_allclose(recovered, forces, atol=1e-3)

    def test_four_thruster_conversion_is_vectorized(self):
        forces = np.array([THRUSTER_FWD_MAX, 0.0, -THRUSTER_REV_MAX, 0.0])

        np.testing.assert_allclose(
            thruster_forces_to_pwm_us(forces),
            [1900.0, 1500.0, 1100.0, 1500.0],
            atol=1e-2,
        )


if __name__ == "__main__":
    unittest.main()
