import unittest

import pytest

pytest.importorskip('ortools', reason='PLL solving requires ortools (pip install hls4ml[nanoxplore])')

from hls4ml.backends.bambu_accelerator.pll_solver import solve_pll

class TestGetAllVcos(unittest.TestCase):
    
    def test_solve_pll_valid_configuration(self):
        """Test with a valid configuration that should produce a solution."""
        input_freq = 400.0
        # These targets can be derived from common VCOs
        targets = [25.0, 100.0, 50.0]
        
        result = solve_pll(input_freq, targets)
        
        self.assertNotIn("Error", result)
        self.assertIn("PLL_", result)
        self.assertIn("NX_PLL_U", result)

    def test_solve_pll_impossible_high_freq(self):
        """Test with a target frequency higher than maximum VCO frequency."""
        input_freq = 400.0
        # Target 900 MHz > VCO MAX (800 MHz)
        targets = [900.0] 
        
        with self.assertRaises(Exception):
            solve_pll(input_freq, targets)

    def test_solve_pll_impossible_low_input_freq(self):
        """Test with input frequency too low to satisfy PFD range."""
        # PFD range is 10-50 MHz.
        # Input 5 MHz. Max PFD = 5/1 = 5 MHz.
        # This is below PFD_MIN (10 MHz).
        input_freq = 5.0
        targets = [100.0]
        
        with self.assertRaises(Exception):
            solve_pll(input_freq, targets)

    def test_solve_pll_empty_targets(self):
        """Test with empty target list."""
        input_freq = 400.0
        targets = []
        
        with self.assertRaises(Exception):
            solve_pll(input_freq, targets)

    def test_solve_pll_complex_configuration(self):
        """Test with the complex configuration from the file example."""
        # This is expected to succeed based on the example in the file.
        input_freq = 400.0
        # Use precise float division for 33.33 MHz to match hardware integer math better or use expected tolerance.
        # 33.33333333 as a literal might miss the exact PFD generation check.
        targets = [25.0, 100.0, 100.0/3.0, 150.0, 0.7]
        
        result = solve_pll(input_freq, targets)
        
        self.assertNotIn("Error", result, "Complex configuration should be solvable")
        self.assertIn("PLL_", result)

    def test_solve_pll_optimization_check(self):
        """Test if the solver optimizes to use minimum PLLs."""
        input_freq = 400.0
        # Targets: 100 and 50.
        # VCO 400 MHz can generate 100 (div 4) and 50 (div 8).
        # Divider 4 and 8 are available.
        # Should use 1 PLL.
        targets = [100.0, 50.0]
        
        result = solve_pll(input_freq, targets)
        
        self.assertIn("PLL_1", result)
        self.assertNotIn("PLL_2", result)

    def test_max_plls_exact(self):
        """Test forcing exactly 7 PLLs."""
        input_freq = 100.0
        # Valid VCOs from 100 MHz input (multiples of 50).
        # We select targets that force specific VCOs and cannot share.
        # Targets:
        # 175 -> VCO 350 (or 700)
        # 225 -> VCO 450
        # 275 -> VCO 550
        # 300 -> VCO 600 (or 900 X)
        # 325 -> VCO 650
        # 375 -> VCO 750
        # 400 -> VCO 800
        # These appear to be mutually exclusive for sharing.
        targets = [175.0, 225.0, 275.0, 300.0, 325.0, 375.0, 400.0]
        
        result = solve_pll(input_freq, targets)
        
        if "Error" in result or "No valid configuration" in result:
             self.fail(f"Solver failed to find solution for 7 valid targets. Result: {result}")
        
        self.assertIn("PLL_7", result)

    def test_max_plls_exceeded(self):
        """Test forcing 8 PLLs (exceeding limit of 7)."""
        input_freq = 100.0
        # Add 166.666... MHz which forces VCO 500 (500/3).
        # It cannot be served by the other 7 VCOs selected above.
        targets = [175.0, 225.0, 275.0, 300.0, 325.0, 375.0, 400.0, 166.6666666667]
        
        with self.assertRaises(Exception):
            solve_pll(input_freq, targets)

    def test_max_outputs_single_pll(self):
        """Test filling all 9 output ports of a single PLL."""
        input_freq = 100.0
        # VCO = 600 MHz
        # Dynamic Ratios (5 slots): 2, 4, 6, 8, 10
        # Static Ratios (4 slots): 3 (S1), 5 (S2), 7 (S3), 9 (S4)
        vco = 600.0
        
        # Ratios available:
        # Dyn: 2, 4, 6, 8, 10
        # Stat: 3, 5, 7, 9
        # Note: 5 is available in both, but we need 9 distinct assignments.
        # We need to pick targets that map to specific ports.
        # Targets:
        t_dyn = [vco/2, vco/4, vco/6, vco/8, vco/10] # 300, 150, 100, 75, 60
        t_stat = [vco/3, vco/5, vco/7, vco/9] # 200, 120, 85.71428, 66.6666
        
        targets = t_dyn + t_stat
        
        result = solve_pll(input_freq, targets)
        
        self.assertNotIn("Error", result)
        # Should fit in 1 PLL
        self.assertIn("PLL_1", result)
        self.assertNotIn("PLL_2", result)

    def test_outputs_exceeded_single_pll_split(self):
        """Test 10 outputs that fit on one VCO frequency but exceed port count, forcing 2 PLLs."""
        input_freq = 100.0
        # VCO = 600 MHz
        # Same as above but add one more dynamic ratio: 20 -> 30 MHz
        vco = 600.0
        t_dyn = [vco/2, vco/4, vco/6, vco/8, vco/10] 
        t_stat = [vco/3, vco/5, vco/7, vco/9]
        t_extra = [vco/20] # 30 MHz
        
        targets = t_dyn + t_stat + t_extra
        
        result = solve_pll(input_freq, targets)
        
        self.assertNotIn("Error", result)
        # Should require 2 PLLs because strict port limit is 9 per PLL
        self.assertIn("PLL_1", result)
        self.assertIn("PLL_2", result)

    def test_solve_pll_375_to_50_register_semantics(self):
        """Reconstruct the output frequency from the emitted registers under the
        hardware-validated encoding (golden top_parallel.v): ref_intdiv = ratio,
        fbk ratio = 2*(fbk_intdiv+1), static S1 divider ratio = 2*code+3."""
        import re
        block = solve_pll(375.0, [50.0])
        ref = int(re.search(r"ref_intdiv\s*\(5'd(\d+)\)", block).group(1))
        fbk = int(re.search(r"fbk_intdiv\s*\(7'd(\d+)\)", block).group(1))
        out1 = int(re.search(r"clk_outdiv1\s*\(3'd(\d+)\)", block).group(1))
        pfd = 375.0 / ref
        vco = pfd * 2 * (fbk + 1)
        out = vco / (2 * out1 + 3)   # S1 static map: ratio = 2*code + 3
        self.assertAlmostEqual(out, 50.0, places=3)

if __name__ == '__main__':
    unittest.main()
