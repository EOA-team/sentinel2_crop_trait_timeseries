# Author: Flavian Tschurr
# Project: KP030
# Date: 21.03.2023
# Purpose:  LAI dose response: response curves
################################################################################




###############################################################################
# response functions
################################################################################

non_linear_response <- function(env_variate,base_value,slope){
  #'@param env_variate value of a environmental covariate
  #'@param base_value estimated value, start of the linear growing phase
  #'@param slope estimated value, slope of the linear phase
  #'@description broken stick modell according to an env variable
  y = (env_variate-base_value)*slope
  y = ifelse(env_variate>base_value,y,0)
  return(y)
}

################################################################################
# looper functions
################################################################################

non_linear_response_loop <- function(env_data_measure, params){
  #'@param env_data_measure vector with environmental measurements (i.e. temp)
  #'@param base_value estimated value, start of the linear growing phase
  #'@param slope estimated value, slope of the linear phase
  #'@description loops over the non linear (broken stick) modell for all given temperature and sums value up)
  #non linear resopnse funtion
  base_value = params[which(names(params)=="base_value")]
  slope = params[which(names(params)=="slope_value")]
  
  growth_modelled_all <-  sum(unlist(lapply(na.omit(env_data_measure), non_linear_response, base_value=base_value,slope=slope)))
  return(growth_modelled_all)
}

non_linear_prediction <- function(env_variate,params){
  #'@param env_variate value of a environmental covariate
  #'@param base_value estimated value, start of the linear growing phase
  #'@param slope estimated value, slope of the linear phase
  #'@description broken stick modell according to an env variable
  #'
  base_value = as.numeric(params[which(names(params)=="base_value")])
  slope = as.numeric(params[which(names(params)=="slope_value")])
  
  y = (env_variate-base_value)*slope
  y = ifelse(env_variate>base_value,y,0)
  return(y)
}


################################################################################
# asymptotic function
################################################################################


# the simple one is not flexible enough 
asymptotic_response <- function(x, Asym, lrc, c0){
  #'@param x input variable
  #'@param Asym  a numeric parameter representing the horizontal asymptote on the right side (very large values of input).
  #'@param lrc 	a numeric parameter representing the natural logarithm of the rate constant.
  #'@param c0  	a numeric parameter representing the x for which the response is zero.
  #'
  
  y <- SSasympOff(x , Asym, lrc, c0)
  y <- ifelse(y > 0, y,0) # no negative growth
  return(y)
}

asymptotic_response_loop <- function(env_data_measure, params){
  #'@param env_data_measure vector with environmental measurements (i.e. temp)
  #'@param base_value estimated value, start of the linear growing phase
  #'@param slope estimated value, slope of the linear phase
  #'@description loops over the non linear (broken stick) modell for all given temperature and sums value up)
  #non linear resopnse funtion
  Asym = params[which(names(params)=="Asym_value")]
  lrc = params[which(names(params)=="lrc_value")]
  c0 = params[which(names(params)=="c0_value")]
  
  
  growth_modelled_all <-  sum(unlist(lapply(na.omit(env_data_measure), asymptotic_response, Asym = Asym, lrc = lrc, c0 = c0)))
  return(growth_modelled_all)
}


# the simple one is not flexible enough 
asymptotic_prediction <- function(x, params){
  #'@param x input variable
  #'@param Asym  a numeric parameter representing the horizontal asymptote on the right side (very large values of input).
  #'@param lrc 	a numeric parameter representing the natural logarithm of the rate constant.
  #'@param c0  	a numeric parameter representing the x for which the response is zero.
  #'
  Asym = as.numeric(params[which(names(params)=="Asym_value")])
  lrc = as.numeric(params[which(names(params)=="lrc_value")])
  c0 = as.numeric(params[which(names(params)=="c0_value")])
  
  y <- SSasympOff(x , Asym, lrc, c0)
  y <- ifelse(y > 0, y,0) # no negative growth
  return(y)
}


asymptotic_constraint <- function(starting_params){
  if(length(which(is.na(as.numeric(starting_params))==T))!=0){
    return(FALSE)
  }
  
  if(as.numeric(starting_params[which(names(starting_params) == "c0_value")]) < as.numeric(starting_params[which(names(starting_params) == "Asym_value")])){
    return(TRUE)  # Constraint satisfied
  } else {
    return(FALSE) # Constraint violated
  }

}


################################################################################
# Wang Engels function
################################################################################



WangEngels_response <- function(x, xmin, xopt,xmax){
  #'@param x effective env_variable value
  #'@param xmin minimal env_variable value of the wang engels model
  #'@param xopt optimal env_variable value according to wang engel model
  #'@param xmax maximal env_variable value according to the wang engel model
  #'
  alpha <- log(2)/(log((xmax-xmin)/(xopt-xmin)))
  
  if(xmin <= x & x <= xmax){
    
    y <- (2*(x-xmin)^alpha*(xopt-xmin)^alpha-(x-xmin)^(2*alpha))/((xopt-xmin)^(2*alpha))
    
  }else{
    y <- 0
  }
  
  return(y)
  
}


WangEngels_response_loop <- function(env_data_measure, params){
  #'@param env_data_measure vector with environmental measurements (i.e. temp)
  #'@param base_value estimated value, start of the linear growing phase
  #'@param slope estimated value, slope of the linear phase
  #'@description loops over the non linear (broken stick) modell for all given temperature and sums value up)
  #non linear resopnse funtion
  xmin = params[which(names(params)=="xmin_value")]
  xopt = params[which(names(params)=="xopt_value")]
  xmax = params[which(names(params)=="xmax_value")]
  
  
  growth_modelled_all <-  sum(unlist(lapply(na.omit(env_data_measure), WangEngels_response, xmin = xmin, xopt = xopt, xmax = xmax)))
  return(growth_modelled_all)
}


WangEngels_prediction <- function(x, params){
  #'@param x effective env_variable value
  #'@param xmin minimal env_variable value of the wang engels model
  #'@param xopt optimal env_variable value according to wang engel model
  #'@param xmax maximal env_variable value according to the wang engel model
  #'
  
  xmin = as.numeric(params[which(names(params)=="xmin_value")])
  xopt = as.numeric(params[which(names(params)=="xopt_value")])
  xmax = as.numeric(params[which(names(params)=="xmax_value")])
  
  
  alpha <- log(2)/(log((xmax-xmin)/(xopt-xmin)))
  
  if(xmin <= x & x <= xmax){
    
    y <- (2*(x-xmin)^alpha*(xopt-xmin)^alpha-(x-xmin)^(2*alpha))/((xopt-xmin)^(2*alpha))
    
  }else{
    y <- 0
  }
  
  return(y)
  
}

WangEngels_constraint <- function(starting_params){
  # if (starting_params$xmin_value < starting_params$xopt_value && starting_params$xopt_value < starting_params$xmax_value) {
  if(length(which(is.na(as.numeric(starting_params))==T))!=0){
    return(FALSE)
  }
  if(as.numeric(starting_params[which(names(starting_params) == "xmin_value")]) < as.numeric(starting_params[which(names(starting_params) == "xopt_value")]) &&
     as.numeric(starting_params[which(names(starting_params) == "xopt_value")]) < as.numeric(starting_params[which(names(starting_params) == "xmax_value")])){
    return(TRUE)  # Constraint satisfied
  } else {
    return(FALSE) # Constraint violated
  }

}


################################################################################
## non linear
################################################################################
estimate_starting_params_non_linear_model <- function(env_vect){
  #'@param env_vect vector with environmental covariate in it
  
  env_vect_sub <- na.omit(env_vect)
  # base value 
  
  base_value <- as.numeric(quantile(env_vect_sub, probs=c(0.05,0.1,0.6)))
  if(base_value[1] == base_value[2]){
    base_value[2] = base_value[2]+0.01
  }
  if(base_value[2] == base_value[3]){
    base_value[3] = base_value[3]+0.01
  }
  
  base_value_list <- list("lower_bound"=base_value[1],"start_value"=base_value[2],"upper_bound"=base_value[3])
  
  slope_value_list <- list("lower_bound"=0,"start_value"=0.05,"upper_bound"=0.5)
  # if(env_variable == "tasmax_2m"){
  #   slope_value_list <- list("lower_bound"=0,"start_value"=0.05,"upper_bound"=0.8)
  #   
  # }
  return(list("base_value"=base_value_list,"slope_value"=slope_value_list))
}



################################################################################
## asymptotic
################################################################################

estimate_starting_params_asymptotic_model <- function(env_vect){
  #'@param env_vect vector with environmental covariate in it
  env_vect_sub <- na.omit(env_vect)
  # base value 
  
  c0_value <- as.numeric(quantile(env_vect_sub, probs=c(0.01,0.05,0.4)))
  if(c0_value[1] == c0_value[2]){
    c0_value[2] = c0_value[2]+0.01
  }
  if(c0_value[2] == c0_value[3]){
    c0_value[3] = c0_value[3]+0.01
  }
  
  c0_value_list <- list("lower_bound"=c0_value[1],"start_value"=c0_value[2],"upper_bound"=c0_value[3])
  
  # lrc_list <- list("lower_bound"=0,"start_value"=4,"upper_bound"=12)
  lrc_list <- list("lower_bound"=-15,"start_value"=-1,"upper_bound"=1.5)
  
  # Asym_value <- as.numeric(quantile(env_vect_sub, probs=c(0.2,0.5,0.99)))
  Asym_value <- as.numeric(quantile(env_vect_sub, probs=c(0.6,0.9,0.98)))
  
  if(Asym_value[1] == Asym_value[2]){
    Asym_value[2] = Asym_value[2]+0.01
  }
  if(Asym_value[2] == Asym_value[3]){
    Asym_value[3] = Asym_value[3]+0.01
  }
  
  Asym_list <- list("lower_bound"=Asym_value[1],"start_value"=Asym_value[2],"upper_bound"=Asym_value[3])
  
  return(list("c0_value"=c0_value_list,"lrc_value"=lrc_list, "Asym_value" = Asym_list))
  
}


################################################################################
## WangEngels
################################################################################

estimate_starting_params_WangEngels_model <- function(env_vect){
  #'@param env_vect vector with environmental covariate in it
  
  env_vect_sub <- na.omit(env_vect)
  # base value 
  
  xmin_value <- as.numeric(quantile(env_vect_sub, probs=c(0.1,0.25,0.4)))
  if(xmin_value[1] == xmin_value[2]){
    xmin_value[2] = xmin_value[2]+0.01
  }
  if(xmin_value[2] == xmin_value[3]){
    xmin_value[3] = xmin_value[3]+0.01
  }
  
  xmin_value_list <- list("lower_bound"=xmin_value[1],"start_value"=xmin_value[2],"upper_bound"=xmin_value[3])
  
  
  xopt_value <- as.numeric(quantile(env_vect_sub, probs=c(0.6,0.85,0.98)))
  if(xopt_value[1] == xopt_value[2]){
    xopt_value[2] = xopt_value[2]+0.01
  }
  if(xopt_value[2] == xopt_value[3]){
    xopt_value[3] = xopt_value[3]+0.01
  }
  
  xopt_value_list <- list("lower_bound"=xopt_value[1],"start_value"=xopt_value[2],"upper_bound"=xopt_value[3])
  
  xmax_value <- as.numeric(quantile(env_vect_sub, probs=c(0.7,0.95,0.99)))
  if(xmax_value[1] == xmax_value[2]){
    xmax_value[2] = xmax_value[2]+0.01
  }
  if(xmax_value[2] == xmax_value[3]){
    xmax_value[3] = xmax_value[3]+0.01
  }
  
  xmax_value_list <- list("lower_bound"=xmax_value[1],"start_value"=xmax_value[2],"upper_bound"=xmax_value[3])
  
  
  return(list("xmin_value"=xmin_value_list,"xopt_value"=xopt_value_list, "xmax_value" = xmax_value_list))
  
}



################################################################################
# function# 
###############################################################################



looper_model_fit_function <- function(rep,
                                      one_measurement_unit,
                                      env_variable, 
                                      parameter_list,
                                      .response_function.,
                                      .constraint_function.,
                                      random_smaple_size = 0.8){
  #'@param one_measurement_unit dataframe containing values of one measurement unit (e.g. plot, genotype etc...)
  #'@param env_variable environmental varaible (eg. temp)
  #'@param parameter_list list with all input aprameters
  #'@param parameters_method decide wheter you want to erstimate the first guess of the input parameters or not (if yes no iterations will be done but 100% of the data will be used --> faster process)
  #'@param .response_function. a function, which will be used to optimize (non_linear, negative_quadratic etc.)
  
  # require(parallel)
  # require(rgenoud)
  source(paste0(getwd(),"/functions/FUN_dose_response_fitting.R"))
  # make fitting more robust: run over 80 percent over the data, 20 times get median of this
  # set iterations and samplesize
  
  # if we fit the modell, we take 20 repetitions with 80 percent of the data each repetitions
  # repetitions <- c(1:20)
  # random_smaple_size <- 0.8
  
  get_boundaries <- function(parameters_numeric, deviation_ratio){
    lower <- NULL
    upper <- NULL
    for(par in 1 :length(parameters_numeric)){
      deviation <- abs(parameters_numeric[par]) * (deviation_ratio)
      lower[par] <- parameters_numeric[par] - deviation
      upper[par] <- parameters_numeric[par] + deviation
    }
    return(list(lower= lower, start = parameters_numeric, upper = upper ))
  }
  
  
  # extract parameters from parameter list
  lower_bounds <- lapply(parameter_list, "[[",1)
  starting_params <- lapply(parameter_list, "[[",2)
  upper_bounds <- lapply(parameter_list, "[[",3)
  names_params <- names(starting_params)
  
  opt_df <- list()
  
  

  random_sample <- sample(x=c(1:length(one_measurement_unit)), size = ceiling(length(one_measurement_unit)*random_smaple_size))
  measurement_list <- one_measurement_unit[random_sample]
  measurement_list <- lapply(measurement_list, function(x) x[!is.na(x)])
  
  
  control_data <- as.numeric(names(measurement_list))

  
  require(nloptr)
  xtol_rel <- 1e-08 # Tolerance criterion between two iterations
  # (threshold for the relative difference of
  # parameter values between the 2 previous
  # iterations)
  maxeval <- 600*length(starting_params) # Maximum number of evaluations of the
  # minimized criteria

  opt_df <- nloptr::auglag(x0 = as.numeric(starting_params),
                           fn = fit_dose_response_model_course,
                           env_data = measurement_list,
                           control_data = control_data,
                           names_params = names_params,
                           .response_function. = .response_function.,
                           .constraint_function. = .constraint_function.,
                           
                           localsolver = c("COBYLA"),
                           localtol =  1e-24,
                
                           lower = as.numeric(lower_bounds),
                           upper= as.numeric(upper_bounds),
                           control = list(
                             xtol_rel = xtol_rel,
                             ftol_rel=  1e-24,
                             ftol_abs =  1e-24,
                             # check_derivatives = TRUE,
                             maxeval = maxeval
                           )
  )



  opt_df[[paste0("mean_",env_variable)]] <- as.numeric(unlist(lapply(measurement_list, mean)))
  names(opt_df$par) <- names_params
  opt_df$growth_modelled <- as.numeric(unlist(lapply(measurement_list, .response_function., opt_df$par)))
  opt_df$growth_measured <- control_data
  
  
  plot(opt_df$growth_modelled, opt_df$growth_measured)
  abline(a=0,b=1)
  cor(opt_df$growth_modelled, opt_df$growth_measured)
  calc_RMSE(opt_df$growth_modelled, opt_df$growth_measured)
  
  if(opt_df$convergence >= 4){
    
    opt_df$par <- rep(NA,length(starting_params))
  }
  
  rm(random_sample, measurement_list)
  
  
  return(opt_df)
  
}




#' ##################################################################################
#' looper_model_fit_function_original <- function(rep,
#'                                       one_measurement_unit,
#'                                       env_variable, 
#'                                       parameter_list,
#'                                       .response_function.,
#'                                       random_smaple_size = 0.8){
#'   #'@param one_measurement_unit dataframe containing values of one measurement unit (e.g. plot, genotype etc...)
#'   #'@param env_variable environmental varaible (eg. temp)
#'   #'@param parameter_list list with all input aprameters
#'   #'@param parameters_method decide wheter you want to erstimate the first guess of the input parameters or not (if yes no iterations will be done but 100% of the data will be used --> faster process)
#'   #'@param .response_function. a function, which will be used to optimize (non_linear, negative_quadratic etc.)
#'   
#'   # require(parallel)
#'   # require(rgenoud)
#'   source(paste0(getwd(),"/functions/FUN_dose_response_fitting.R"))
#' 
#'   # make fitting more robust: run over 80 percent over the data, 20 times get median of this
#'   # set iterations and samplesize
#'   
#'   # if we fit the modell, we take 20 repetitions with 80 percent of the data each repetitions
#'   # repetitions <- c(1:20)
#'   # random_smaple_size <- 0.8
#'   
#'   get_boundaries <- function(parameters_numeric, deviation_ratio){
#'     lower <- NULL
#'     upper <- NULL
#'     for(par in 1 :length(parameters_numeric)){
#'       deviation <- abs(parameters_numeric[par]) * (deviation_ratio)
#'       lower[par] <- parameters_numeric[par] - deviation
#'       upper[par] <- parameters_numeric[par] + deviation
#'     }
#'     return(list(lower= lower, start = parameters_numeric, upper = upper ))
#'   }
#'   
#'   
#'   # extract parameters fromparameter list
#'   lower_bounds <- lapply(parameter_list, "[[",1)
#'   starting_params <- lapply(parameter_list, "[[",2)
#'   upper_bounds <- lapply(parameter_list, "[[",3)
#'   names_params <- names(starting_params)
#'   
#'   opt_df <- list()
#'   
#'   
#'   # for(rep in repetitions){
#'   
#'   random_sample <- sample(x=c(1:length(one_measurement_unit)), size = ceiling(length(one_measurement_unit)*random_smaple_size))
#'   measurement_list <- one_measurement_unit[random_sample]
#'   measurement_list <- lapply(measurement_list, function(x) x[!is.na(x)])
#'   
#'   
#'   # control_data <- rep(100,length(names(measurement_list)))
#'   control_data <- as.numeric(names(measurement_list))
#' 
#'   first_optim <- try(optim(par = as.numeric(starting_params),
#'                            fn = fit_dose_response_model_course,
#'                            env_data = measurement_list,
#'                            control_data = control_data,
#'                            names_params = names_params,
#'                            .response_function. = .response_function.,
#'                            method = "BFGS", hessian = T,
#'                            control = list(trace = 1,
#'                                           maxit =40,
#'                                           ndeps = rep(1,length(starting_params)))), silent = T)
#'   
#'   
#'   
#'   
#'   
#'   first_nlm <- nlm(f = fit_dose_response_model_course,
#'                    p = as.numeric(starting_params),
#'                    env_data = measurement_list,
#'                    control_data = control_data,
#'                    names_params = names_params,
#'                    .response_function. = .response_function.,
#'                    hessian = T,
#'                    iterlim = 50,
#'                    steptol = 1e-2)
#'   
#'   # find better method
#'   methods <- c("optim","nlm")
#'   if(class(first_optim) == "list"){
#'     method = methods[which.min(c(first_optim$value,first_nlm$minimum))]
#'     if(first_nlm$code > 1){
#'       method = "optim"
#'     }
#'     
#'     
#'   }else{
#'     method = "nlm"
#'     if(first_nlm$code> 3){
#'       first_nlm <- nlm(f = fit_dose_response_model_course,
#'                        p = as.numeric(starting_params),
#'                        env_data = measurement_list,
#'                        control_data = control_data,
#'                        names_params = names_params,
#'                        .response_function. = .response_function.,
#'                        hessian = T,
#'                        iterlim = 100)
#'       
#'     }
#'   }
#'   
#'   if(method == "optim"){
#'     next_params <- get_boundaries(first_optim$par,deviation_ratio = 1)
#'     
#'   }else if( method == "nlm"){
#'     next_params <- get_boundaries(first_nlm$estimate,deviation_ratio = 1)
#'   }
#'   
#'   # 
#'   # # second partial optimization without boundaries
#'   # second_opt <- try(optim(par = as.numeric(next_params$start),
#'   #                         fn = fit_dose_response_model_course,
#'   #                         env_data = measurement_list,
#'   #                         control_data = control_data,
#'   #                         names_params = names_params,
#'   #                         .response_function. = .response_function.,
#'   #                         lower = as.numeric(next_params$lower),
#'   #                         upper = as.numeric(next_params$upper),
#'   #                         method = "L-BFGS-B",
#'   #                         hessian = T,
#'   #                         control = list(trace = 1,
#'   #                                        maxit = 100,
#'   #                                        ndeps = rep(1e-1,length(starting_params)))),silent=TRUE)
#'   # 
#'   # 
#'   # if(class(second_opt) == "try-error" ){
#'   #   rm(second_opt)
#'   #   second_opt <- nlm(f = fit_dose_response_model_course,
#'   #                     p = as.numeric(next_params$start),
#'   #                     env_data = measurement_list,
#'   #                     control_data = control_data,
#'   #                     names_params = names_params,
#'   #                     .response_function. = .response_function.,
#'   #                     hessian = T,
#'   #                     iterlim = 50)
#'   #   
#'   #   second_opt$convergence <- ifelse(second_opt$code <=2 ,0, 1)
#'   #   second_opt$par <- second_opt$estimate
#'   #   
#'   # }
#'   # if(second_opt$convergence != 0){
#'   #   second_opt <- nlm(f = fit_dose_response_model_course,
#'   #                     p = as.numeric(next_params$start),
#'   #                     env_data = measurement_list,
#'   #                     control_data = control_data,
#'   #                     names_params = names_params,
#'   #                     .response_function. = .response_function.,
#'   #                     hessian = T,
#'   #                     iterlim = 100)
#'   #   
#'   #   second_opt$par <- second_opt$estimate
#'   # }
#'   # # calculate SE or use a ratio (0.5) and optimize with BBoptim
#'   # 
#'   # hessian.inv <- try(solve(second_opt$hessian),silent = T)
#'   # if(any(class(hessian.inv) == "matrix")){
#'   #   parameter.se <- sqrt(diag(abs(hessian.inv)))
#'   #   next_params <- list( "lower" = second_opt$par -parameter.se,
#'   #                        "start" = second_opt$par,
#'   #                        "upper" = second_opt$par + parameter.se)
#'   #   
#'   # }else{
#'   #   next_params <- get_boundaries(second_opt$par,deviation_ratio = 0.5)
#'   #   
#'   # }
#'   # 
#'   
#'   
#'   require(BB)
#'   
#'   opt_df <- BB::BBoptim(par = as.numeric(next_params$start),
#'                         fn = fit_dose_response_model_course,
#'                         env_data = measurement_list,
#'                         control_data = control_data,
#'                         names_params = names_params,
#'                         .response_function. = .response_function.,
#'                         method = c(3,2,1),
#'                         lower = as.numeric(next_params$lower),
#'                         upper = as.numeric(next_params$upper),
#'                         control = list(trace = 1,
#'                                        maxit=(100*length(next_params$start))))
#'   
#' 
#'   opt_df[[paste0("mean_",env_variable)]] <- as.numeric(unlist(lapply(measurement_list, mean)))
#'   names(opt_df$par) <- names_params
#'   opt_df$growth_modelled <- as.numeric(unlist(lapply(measurement_list, .response_function., opt_df$par)))
#'   opt_df$growth_measured <- control_data
#'   
#'   if(opt_df$convergence != 0){
#'     
#'     opt_df$par <- rep(NA,length(starting_params))
#'   }
#'   
#'   rm(random_sample, measurement_list)
#' 
#'   
#'   return(opt_df)
#'   
#' }
#' 


################################################################################
# fitting functions
################################################################################


# function to fit 2 param model
fit_dose_response_model_course <- function(params,env_data,control_data,names_params, .response_function., .constraint_function.){
  # fit_dose_response_model_course <- function(params,argument_list){
  
  #'@param params list with paramters (per paramter 3 values, min, start and max)
  #'@param env_data list with envrionmental data (per measurement a vector in the list)
  #'@param names_params names of the paramters
  #'@param .response_function. response function 
  
  # get parameters
  source("functions/FUN_skillscores.R")
  # name the aprameteres in the vector accordingly
  names(params) <- names_params
  print(params)
  if(!is.null(.constraint_function.)){
    if(.constraint_function.(params)==FALSE){
      skill_score <- 10000
      print("constraint")
      return(skill_score)
    }
  }
  
  # get values
  growth_modelled <- unlist(lapply(env_data, .response_function., params))
  # calculate skill score
  
  
  skill_score <- calc_RMSE(measured = control_data, modelled = growth_modelled)
  # skill_score <- calc_SumLogLikelihood(measured = control_data, modelled = growth_modelled,sigma_error = 10)
  
  if(length(which(round(growth_modelled,10)==0)) >= length(growth_modelled)*0.75){
    # skill_score <- 10000
    print("low values")
    skill_score <- skill_score * 100
  }
  if(is.na(skill_score)){
    skill_score <- 10000
  }
 
  return(skill_score)
  
}


