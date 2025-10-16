The proiject has three main files: 

 --- The Kriging Kneading.txt file is the dataset found originally in the paper by Ismail et al (2019) found at https://doi.org/10.1016/j.powtec.2018.11.060
 --- The NARX__univariates_trainbr.m is the NARX ANN model coupled with various univariate interpolation methods and using the trainbr
 --- The NARX_Multivariate_cubicspline.m file is the model using the multivariate cubic spline method with NARX ANN and trainbr training algorithm


 --- The additional files were also used in the paper but were not the main prevailing codes of the successful models. I have included here only for your interest:
 --- The NARX_Univariates_trainlm.m is the same as the above NARX__univariates_trainbr.m file but uses the trainlm algorithm and has been architectually optimised for that algorithm.
 --- The NARX_Kriging.m file is my replication of the results presented in the Ismail et al (2019) paper above using the Kriging interpolation method. 
