1) Environment: https://drive.google.com/drive/folders/128E_wL2bBFVVITpMRrhShk5oRLy02mUF?usp=sharing  
    Description: This is plain environment files without Airsim plugin

2) Scaled drone: https://drive.google.com/drive/folders/1AMhjyyjzijP1EcZYIISuqwbYf8bF7N_G?usp=sharing  
    Description: The default drone mesh size is scaled down by factor of 10 to make it more compatible with the environment.  
        Issues: Propellers out or position, PID might not be right.

3) Environment scaled with collisions disabled: https://drive.google.com/file/d/1DZb-bfjybF1xg585k2GtYo4idxeW7idH/view?usp=sharing  
    Description: The default drone mesh is conserved and the environment is scaled by a factor of 10(in terms of mesh size, the real world dimensions are conserved hence maintaining the physics). Collisions are disabled in this environment.

4) Environment with custom drone: https://drive.google.com/file/d/1WpVjCEwOvmiTE-4BTEsTxE8PD7iPbbxF/view?usp=sharing
    Description: The custom drone is added to the environment.  
        Usage: Go to the Documents/AirSim/settings.json and replace it with the settings.json in this repository
