#import the packages you need
import numpy as np
import copy

class FinitDifferenceAdvection(object):
    def __init__(self, phi=None, delta_x=None, delta_t=None, u_0=None):
        self.phi = phi
        self.delta_x = delta_x
        self.delta_t = delta_t
        self.u_0 = u_0
        self.mu = self.u_0 * self.delta_t / self.delta_x
        self.mu_stable = self.mu < 1
        # CFL condition 
        print("For a courant number equal to be less than 1 (our limit) delta_t < %.2f" %((self.delta_x/self.u_0)))
        print("Number of steps required for 15 days %.2f" % (15 * 86400 / delta_t))              
        print("Actual: Courant number %.2f - Stable %s" %(self.mu, self.mu_stable))
        
    
    def stability_check(self, method_name: str, phi: np.array, max_phi: float, total_phi: float):
        # Asserts
        assert method_name in ["phi_max", "phi_total"]
        # Dispatcher
        method = getattr(self, method_name, lambda: "Not implemented")
        return method(phi, max_phi, total_phi)
    
    def phi_max(self, phi: np.array, max_phi: float, total_phi: float):
        return [np.abs(max_phi - np.abs(phi).max()), np.abs(phi).max()]
    
    def phi_total(self, phi: np.array, max_phi: float, total_phi: float):
        return [np.abs(total_phi - phi.sum()), phi.sum()]
    
    def fd_solver(self, method_name: str, technique: str, t_max: float, error: bool):
        # Create a temporal array -> for swaping data
        phi_temp = np.zeros_like(self.phi)
        if error:
            phi = copy.deepcopy(self.phi) + np.random.normal(0, 0.001, len(self.phi))
        else:
            phi = copy.deepcopy(self.phi)        
        # Asserts
        assert (self.phi is not None), "Field phi is empty in object"
        assert (self.delta_x is not None), "Variable delta_x not defined in object"
        assert (self.delta_t is not None), "Variable delta_t not defined in object"
        assert (self.u_0 is not None), "Variable u_0 not defined in object"
        assert method_name in ["ftus", "ftds", "ftcs", "leapfrog"]
        assert technique in ["direct", "numpy", "roll"]
        # Some relevatn variable
        c = self.u_0 * self.delta_t / self.delta_x
        n_time = int(t_max / self.delta_t)
        max_phi = phi.max()
        total_phi = phi.sum()
        # Dispatcher
        method = getattr(self, method_name + "_" + technique, lambda: "Not implemented")
        return method(phi, phi_temp, c, n_time, max_phi, total_phi)
    
    def ftus_direct(self, phi: np.array, phi_temp: np.array, c: float, n_time: int, max_phi: float, total_phi: float):
        # Let's avoid unncesary divisions and replace by multiplications
        error_growth = []
        for t in range(0, n_time):
            for i in range(0, phi.size): 
                phi_temp[i] = phi[i] - c * (phi[i]-phi[i-1])
            phi[:] = phi_temp[:]
            #TODO: We need a stability check function
            error_growth.append(np.abs(max_phi - np.abs(phi).max()))
            if (np.abs(phi).max() > 2 * max_phi):
                print("Simulation unstable at step %.d" % (t))
                break;
        return phi, error_growth

    def ftus_numpy(self, phi: np.array, phi_temp: np.array, c: float, n_time: int, max_phi: float, total_phi: float):
        # Let's avoid unncesary divisions and replace by multiplications
        error_growth = []
        for t in range(0, n_time):
            phi_temp[1:] = phi[1:] - c * (phi[1:] - phi[:-1])
            phi_temp[0] = phi[0] - c * (phi[0] - phi[-1])
            phi[:] = phi_temp[:]
            #TODO: We need a stability check function
            error_growth.append(np.abs(max_phi - np.abs(phi).max()))
            if (np.abs(phi).max() > 2 * max_phi):
                print("Simulation unstable at step %.d" % (t))
                break;
        return phi, error_growth
    
    def ftus_roll(self, phi: np.array, phi_temp: np.array, c: float, n_time: int, max_phi: float, total_phi: float):
        # Let's avoid unncesary divisions and replace by multiplications
        error_growth = []
        for t in range(0, n_time):
            phi[:] = phi[:] - c * (phi[:] - np.roll(phi[:],-1))
            #TODO: We need a stability check function
            error_growth.append(np.abs(max_phi - np.abs(phi).max()))
            if (np.abs(phi).max() > 2 * max_phi):
                print("Simulation unstable at step %.d" % (t))
                break;
        return phi, error_growth
    
    def ftds_direct(self, phi: np.array, phi_temp: np.array, c: float, n_time: int, max_phi: float, total_phi: float):
        # Let's avoid unncesary divisions and replace by multiplications
        error_growth = []
        for t in range(0, n_time):
            for i in range(0, phi.size):
                if (i == phi.size - 1):
                    phi_temp[i] = phi[i] - c * (phi[0]-phi[i])
                else:
                    phi_temp[i] = phi[i] - c * (phi[i+1]-phi[i])
            phi[:] = phi_temp[:]
            #TODO: We need a stability check function
            error_growth.append(np.abs(max_phi - np.abs(phi).max()))
            if (np.abs(phi).max() > 2 * max_phi):
                print("Simulation unstable at step %.d" % (t))
                break;
        return phi, error_growth
    
    def ftds_numpy(self, phi: np.array, phi_temp: np.array, c: float, n_time: int, max_phi: float, total_phi: float):
        # Let's avoid unncesary divisions and replace by multiplications
        error_growth = []
        for t in range(0, n_time):
            # numpy trick - way faster
            phi_temp[:-1] = phi[:-1] - c * (phi[1:] - phi[:-1])
            phi_temp[-1] = phi[-1] - c * (phi[0] - phi[-1])
            phi[:] = phi_temp
            #TODO: We need a stability check function
            error_growth.append(np.abs(max_phi - np.abs(phi).max()))
            if (np.abs(phi).max() > 2 * max_phi):
                print("Simulation unstable at step %.d" % (t))
                break;
        return phi, error_growth
    
    def ftds_roll(self, phi: np.array, phi_temp: np.array, c: float, n_time: int, max_phi: float, total_phi: float):
        # Let's avoid unncesary divisions and replace by multiplications
        error_growth = []
        for t in range(0, n_time):
            phi[:] = phi[:] - c * (np.roll(phi[:],1) - phi[:])
            #TODO: We need a stability check function
            error_growth.append(np.abs(max_phi - np.abs(phi).max()))
            if (np.abs(phi).max() > 2 * max_phi):
                print("Simulation unstable at step %.d" % (t))
                break;
        return phi, error_growth
    
    def ftcs_direct(self, phi: np.array, phi_temp: np.array, c: float, n_time: int, max_phi: float, total_phi: float):
        # Let's avoid unncesary divisions and replace by multiplications
        error_growth = []
        for t in range(0, n_time):
            for i in range(0, phi.size):
                if (i == phi.size - 1):
                    phi_temp[i] = phi[i] - c / 2 * (phi[0]-phi[i-1])
                else:
                    phi_temp[i] = phi[i] - c / 2 * (phi[i+1]-phi[i-1])
            phi[:] = phi_temp
            #TODO: We need a stability check function
            error_growth.append(np.abs(max_phi - np.abs(phi).max()))
            if (np.abs(phi).max() > 2 * max_phi):
                print("Simulation unstable at step %.d" % (t))
                break;
        return phi, error_growth
    
    def ftcs_numpy(self, phi: np.array, phi_temp: np.array, c: float, n_time: int, max_phi: float, total_phi: float):
        # Let's avoid unncesary divisions and replace by multiplications
        error_growth = []
        for t in range(0, n_time):
            # numpy trick - way faster
            phi_temp[1:-1] = phi[1:-1] - c / 2 * (phi[2:] - phi[:-2])
            phi_temp[0] = phi[0] - c / 2 * (phi[1] - phi[-1])
            phi_temp[-1] = phi[-1] - c / 2 * (phi[0] - phi[-2])
            phi[:] = phi_temp
            #TODO: We need a stability check function
            error_growth.append(np.abs(max_phi - np.abs(phi).max()))
            if (np.abs(phi).max() > 2 * max_phi):
                print("Simulation unstable at step %.d" % (t))
                break;
        return phi, error_growth
    
    def ftcs_roll(self, phi: np.array, phi_temp: np.array, c: float, n_time: int, max_phi: float, total_phi: float):
        # Let's avoid unncesary divisions and replace by multiplications
        error_growth = []
        for t in range(0, n_time):
            phi[:] = phi[:] - c / 2 * (np.roll(phi[:],1) - np.roll(phi[:],-1))
            #TODO: We need a stability check function
            error_growth.append(np.abs(max_phi - np.abs(phi).max()))
            if (np.abs(phi).max() > 2 * max_phi):
                print("Simulation unstable at step %.d" % (t))
                break;
        return phi, error_growth
    
    def leapfrog_direct(self, phi: np.array, phi_temp: np.array, c: float, n_time: int, max_phi: float, total_phi: float):
        # Let's avoid unncesary divisions and replace by multiplications
        phi_ = copy.deepcopy(phi)
        phi, error_growth = self.ftus_direct(phi, phi_temp, c, 1, max_phi, total_phi)
        for t in range(1, n_time):
            for i in range(0, phi.size):
                if (i == phi.size - 1):
                    phi_temp[i] = phi_[i] - c * (phi[0]-phi[i-1])
                else:
                    phi_temp[i] = phi_[i] - c * (phi[i+1]-phi[i-1])
            phi_[:] = phi[:]
            phi[:] = phi_temp[:]       
            #TODO: We need a stability check function
            error_growth.append(np.abs(max_phi - np.abs(phi).max()))
            if (np.abs(phi).max() > 2 * max_phi):
                print("Simulation unstable at step %.d" % (t))
                break;
        return phi, error_growth
    
    def leapfrog_numpy(self, phi: np.array, phi_temp: np.array, c: float, n_time: int, max_phi: float, total_phi: float):
        # Let's avoid unncesary divisions and replace by multiplications
        phi_ = copy.deepcopy(phi)
        phi, error_growth = self.ftus_direct(phi, phi_temp, c, 1, max_phi, total_phi)        
        for t in range(1, n_time):
            # numpy trick - way faster
            phi_temp[1:-1] = phi_[1:-1] - c * (phi[2:] - phi[:-2])
            phi_temp[0] = phi_[0] - c * (phi[1] - phi[-1])
            phi_temp[-1] = phi_[-1] - c * (phi[0] - phi[-2])
            phi_[:] = phi[:]
            phi[:] = phi_temp[:] 
            #TODO: We need a stability check function
            error_growth.append(np.abs(max_phi - np.abs(phi).max()))
            if (np.abs(phi).max() > 2 * max_phi):
                print("Simulation unstable at step %.d" % (t))
                break;
        return phi, error_growth
    
    def leapfrog_roll(self, phi: np.array, phi_temp: np.array, c: float, n_time: int, max_phi: float, total_phi: float):
        # Let's avoid unncesary divisions and replace by multiplications
        phi_ = copy.deepcopy(phi)
        phi, error_growth = self.ftus_direct(phi, phi_temp, c, 1, max_phi, total_phi)        
        for t in range(1, n_time):
            # numpy trick - way faster
            phi_temp[:] = phi_[:] - c * (np.roll(phi[:],1) - np.roll(phi[:],-1))
            phi_[:] = phi[:]
            phi[:] = phi_temp[:] 
            #TODO: We need a stability check function
            error_growth.append(np.abs(max_phi - np.abs(phi).max()))
            if (np.abs(phi).max() > 2 * max_phi):
                print("Simulation unstable at step %.d" % (t))
                break;
        return phi, error_growth