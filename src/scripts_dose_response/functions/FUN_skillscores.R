# Author: Flavian Tschurr
# Project: KP030
# Date: 13.05.2022
# Purpose: Winterkill: skill scores
################################################################################

calc_RMSE <- function(measured, modelled){
  return(sqrt(mean((measured - modelled)^2)))
}

calc_RMSE_rel <- function(measured, modelled){
  diff_perc <- (modelled-measured)/measured*100
  return( sqrt(mean((diff_perc)^2)))
}

calc_MAE <- function(measured, modelled){
  return(abs(mean(measured - modelled)))
}

calc_MAE_rel <- function(measured, modelled){
  return(abs(mean((measured-modelled)/measured*100)))
}


calc_SumLogLikelihood <- function(measured, modelled, sigma_error=10) {

  if (sigma_error < 0) {
    deviance <- 10000000
  } else {
  
    likelihoods <- dnorm(measured, mean = modelled, sd = sigma_error)
    
    log.likelihoods <- log(likelihoods)
    
    # deviance <- -1 * sum(log.likelihoods)
    deviance <- -2 * sum(log.likelihoods)
    
  }
  
  if (is.infinite(deviance)) {
    deviance <- 10000000
  }
  
  return(deviance)
}


calc_cor <- function(measured, modelled){
  return(cor(measured,modelled,method=c("pearson")))
}

