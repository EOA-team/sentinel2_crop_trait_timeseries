# Author: Flavian Tschurr
# Project: KP030
# Date: 21.03.2023
# Purpose:  LAI dose response: utility functions
################################################################################






sorting_env_to_list <- function(one_measurement_unit, variable, df_env, env_variable){
  #'@param one_measurement_unit df with measurements (needs a timestamp column)
  #'@param variable measurement variable (e.g. delta_CC)
  #'@param df_env environmental covariate (needs a timestamp column)
  #'@param env_variable env variable (e.g. tas_2m)
  #'
  require(dplyr)
  measurements_list <- list()
  for(measure in 2:length(one_measurement_unit[[variable]])){
    start_date <- one_measurement_unit$timestamp[(measure-1)]
    end_date <-  one_measurement_unit$timestamp[measure]
    df_env_subs <-  df_env %>%
      filter(timestamp <  end_date)%>%
      filter(timestamp > start_date)
    if(length(df_env_subs[[env_variable]])== 0){
      # next
      measurements_list[[as.character(one_measurement_unit[[variable]][measure])]] <- NULL
    }else{
      measurements_list[[as.character(one_measurement_unit[[variable]][measure])]] <- df_env_subs[[env_variable]]
    }
    
  }
  
      # browser()
  return(measurements_list)
  
}



combined_data_cleaning_function <- function(one_measurement_unit,
                                            df_env,
                                            variable,
                                            env_variable,
                                            data_cleaning= NA ){
  #'@param one_measurement_unit dataframe containing values of one measurement unit (e.g. plot, genotype etc...)
  #'@param df_env dataframe with environmental data (contains values and a "timestamp" column --> containing the timestamps)
  #'@param variable measurement variable of the one_measurement_unit data frame
  #'@param env_variable env variable (e.g. tas_2m)
  #'@param data_cleaining options how data should be cleaned
  require(dplyr)
  # write all values into a list
  measurement_list <- sorting_env_to_list(one_measurement_unit, variable, df_env, env_variable)
  # data cleaning options
  if("no_negative_values" %in% data_cleaning){
    if(length(which(as.numeric(names(measurement_list)) < 0))== length(names(measurement_list))){
      return(NA)
    }else{
      measurement_list <- remove_negative_values(measurement_list)
    }
  }
  if("select_negative_values" %in% data_cleaning){
    measurement_list <- select_negative_values(measurement_list)
  }
  
  
  return(measurement_list)
}



remove_negative_values <- function(measurement_list){
  #'@param measurement_list list, names = measurements, values within the list are the environmental data
  #'@description deletes all list entries with negative measurement values
  
  to_remove <- NULL
  for(i in 1:length(measurement_list)){
    if(as.numeric(names(measurement_list)[i])< 0){
      # measurements_list <- measurements_list[-i]
      to_remove <- append(to_remove,i)
    }
  }
  
  if(length(to_remove)>0){
    measurement_list <- measurement_list[-to_remove]
  }
  
  return(measurement_list)
  
}


################################################################################

select_negative_values <- function(measurement_list){
  #'@param measurement_list list, names = measurements, values within the list are the environmental data
  #'@description deletes all list entries with negative measurement values
  to_keep<- NULL
  for(i in 1:length(measurement_list)){
    if(as.numeric(names(measurement_list)[i])< 0){
      # measurements_list <- measurements_list[-i]
      to_keep <- append(to_keep,i)
    }
  }
  
  if(length(to_keep)>0){
    measurement_list <- measurement_list[to_keep]
  }
  
  return(measurement_list)
  
}





get_median_of_parameters <- function(one_output){
  #'@param one_output output list of the optimized paramters
  #'@description calculated the median of the given input paramters and returns it 
  if(length(one_output[[1]])==1){
    # browser()
    return(NA)
  }
  params_list <- list()
  for(param in c(1:length(one_output[[1]]$par))){
    out_vect <- NULL
    for (i in 1:length(one_output)) {
      out_vect[i] <- one_output[[i]]$par[param] 
      
    }
    
    params_list[[param]] <- median(out_vect,na.rm = T)
  }
  returner <- unlist(params_list)
  return(unlist(params_list))
  
}


get_min_mean_median_max_of_parameters <- function(one_output){
  #'@param one_output output list of the optimized paramters
  #'@description calculated the median of the given input paramters and returns it 
  
  # browser()
  if(length(one_output[[1]])==1){
    return(NA)
  }
  params_list <- list()
  for(param in c(1:length(one_output[[1]]$par))){
    out_vect <- NULL
    for (i in 1:length(one_output)) {
      out_vect[i] <- one_output[[i]]$par[param] 
      
    }
    params_list[[param]] <- c('min' = min(out_vect,na.rm=T), 'mean' = mean(out_vect,na.rm=T) , 'median' = median(out_vect,na.rm=T) , 'max' = max(out_vect,na.rm=T))
  }
  returner <- unlist(params_list)
  return(unlist(params_list))
  
}


