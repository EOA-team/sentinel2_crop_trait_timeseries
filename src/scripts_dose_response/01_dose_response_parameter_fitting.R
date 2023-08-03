# Author: Flavian Tschurr
# Project: KP030
# Date: 08.03.2022
# Purpose: LAI dose response: Prepare input data + fit dose response curves
################################################################################
path_script_base <- "."
setwd(path_script_base)
#packages
library(dplyr)
library(parallel)
library(lubridate)
source(paste0(path_script_base,"/functions/utils_dose_response.R"))
source(paste0(path_script_base,"/functions/FUN_dose_response_fitting.R"))

# constants
base_path_data <- "../../data/dose_reponse_in-situ"

granularity = "hourly"
# granularity = "daily"

combined_measurement_list <- readRDS(paste0(base_path_data,"/output/LAI_",granularity,"_Bramenwies_MNI_Rur.rds"))

location_id = "CH_Bramenwies"
response_curve_types <- c("non_linear", "asymptotic","WangEngels")

# 
# # script
# ################################################################################
# # read LAI data
# ################################################################################
# LAI <- read.csv(paste(base_path_data,location_id,"LAI_BW_Raw-Data.csv",sep="/"))
# LAI$Point_ID <- as.factor(paste0("p_",LAI$Point_ID))
# 
# LAI$timestamp <- as.POSIXct(LAI$dateonly)
# 
# df_LAI <- LAI %>%
#   mutate(DOY = yday(timestamp)) %>%
#   group_by(Point_ID) %>%
#   arrange(DOY) %>%
#   mutate(LAI_smooth = smooth.spline(x= DOY, y = LAI_value)$y)%>%
#   mutate(lag_DOY = lag(DOY),
#          lag_LAI = lag(LAI_value),
#          delta_LAI = (LAI_value - lag_LAI),
#          lag_LAI_smooth = lag(LAI_smooth),
#          delta_LAI_smooth = (LAI_smooth - lag_LAI_smooth))
# 
# library(ggplot2)
# 
# ggplot(data=df_LAI, aes(timestamp,LAI_smooth, color=Point_ID))+
#   geom_line()+ theme_classic()
# ################################################################################
# # read env data
# ################################################################################
# 
env <- read.csv(paste(base_path_data,location_id,"Meteo_BW.csv",sep="/"))
env$timestamp <- as.POSIXct(env$time)
env <- env %>%
  arrange(timestamp)

################################################################################
# clean and combine data
################################################################################
variable = "delta_LAI_smooth"
# variables <- c("delta_LAI_smooth", "delta_LAI")
env_variable = "T_mean"
# for(variable in variables){
  
  
  # all_points <- list()
  # for(pl in unique(df_LAI$Point_ID)) {
  #   all_points[[pl]] <- df_LAI %>%
  #     filter(Point_ID == pl) #%>%
  # }
  # 
  # measurement_list <-lapply(all_points, 
  #                           combined_data_cleaning_function,
  #                           df_env = env,
  #                           variable=variable,
  #                           env_variable = env_variable,
  #                           data_cleaning="no_negative_values")
  # 
  # combined_measurement_list <- list()
  # predictions<- list()
  # for(point_ in names(measurement_list)){
  #   for(measurement in names(measurement_list[[point_]])){
  #     combined_measurement_list[[measurement]]<-  measurement_list[[point_]][[measurement]]
  #   }
  # }
  #   
  
  for (response_curve_type in response_curve_types) {
    # browser()
    print(paste("start model fitting: ",response_curve_type,sep=" "))
    # catch response function
    .response_function. <- get(paste0(response_curve_type,"_response_loop"))
    
    # get constriant function
    .constraint_function. <- try(get(paste0(response_curve_type,"_constraint")),silent = T)
    if(class(.constraint_function.)== "try-error"){
      .constraint_function. = NULL
    }
    
    # get starting parameter function
    
    .starting_parameter_estimation_function. <- get(paste0("estimate_starting_params_",response_curve_type,"_model"))
    # get starting parameters
    parameter_list <- .starting_parameter_estimation_function.(env_vect = env[[env_variable]])
    # select how many times a random sample shoul be taken to optimize paramters over
    repetitions <- c(1:20)
  
    # paralellization
    numCores <- min(detectCores(),length(repetitions))
    cl <- makePSOCKcluster(numCores)
    
    start_time <- Sys.time()
    


    output <- parallel::parLapplyLB(cl,
                                    repetitions,
                                    looper_model_fit_function,
                                    one_measurement_unit = combined_measurement_list,
                                    env_variable = env_variable,
                                    parameter_list = parameter_list,
                                    .response_function. = .response_function.,
                                    .constraint_function. = .constraint_function.,
                                    random_smaple_size = 0.8)
    
    
    # opt_df <- lapply(repetitions,
    #                                 looper_model_fit_function,
    #                                 one_measurement_unit = combined_measurement_list,
    #                                 env_variable = env_variable,
    #                                 parameter_list = parameter_list,
    #                                 .response_function. = .response_function.,
    #                                 .constraint_function. = .constraint_function.,
    #                                 random_smaple_size = 0.8)
    
    stopCluster(cl)
    
    end_time <- Sys.time()
    print(paste("modelfitting  done for:", response_curve_type,env_variable,sep=" "))
    print(end_time - start_time)
    
    
    # create meta information before saving
    meta_info <- list()
    meta_info[["response_curve_type"]] <- response_curve_type
    meta_info[["env_variable"]] <- env_variable
    meta_info[["parameter_list"]] <- parameter_list
    meta_info[["additional_description"]] <-"20 times optimized over 80% of the data"
    # write into the correct list
  
    # calculate median
    median_output<- list()
    median_output[[paste(env_variable,response_curve_type,sep="-")]] <- get_median_of_parameters(output)
    median_output[["meta"]] <- meta_info
    
    all_statistics_output <- list()
    all_statistics_output[[paste(env_variable,response_curve_type,sep="-")]] <- get_min_mean_median_max_of_parameters(output)
    all_statistics_output[["meta"]] <- meta_info
    
    
    combined_output <- list()
    combined_output[["median_output"]] <- median_output
    combined_output[["complete_output"]] <- output
    combined_output[["parameter_range_output"]] <- all_statistics_output
    # crate directory
    # browser()
    output_path_base <- paste0(base_path_data,"/output/parameter_model/",response_curve_type)
    
    dir.create(output_path_base,recursive = T, showWarnings = F)
  
    # out_file_name <- paste0(response_curve_type,"_variable_",variable,"_parameter_" ,env_variable,"_location_",location_id,".rds")
    out_file_name <- paste0(response_curve_type,"_granularity_",granularity,"_parameter_" ,env_variable,".rds")
    
    
    saveRDS(combined_output, file= paste0(output_path_base,"/",out_file_name))
    
    # apply best parameters to data and calculate correlation etc.
    
    
    .response_function_prediction. <- get(paste0(response_curve_type,"_prediction"))
    parameters <- get_median_of_parameters(output)
    names(parameters) <- names(median_output$meta$parameter_list)
    
    # write just parameters into data.frame
    parameters_df <- data.frame(parameter_name = names(parameters), parameter_value = as.numeric(parameters))
    # out_file_name_parameters <- paste0(response_curve_type,"_variable_",variable,"_parameter_" ,env_variable,"_location_",location_id,".csv")
    out_file_name_parameters <- paste0(response_curve_type,"_granularity_",granularity,"_parameter_" ,env_variable,".csv")
    
    write.csv(parameters_df, file= paste0(output_path_base,"/",out_file_name_parameters))
    
    # lapply(combined_measurement_list, .response_function_prediction., parameters)
    
    # debug not paralellized
    # predictions[[env_variable]][["modelled"]]<- lapply(pheno_list, pred_helper, .response_function., parameters,weight_parameter, granularity)
    # granularity <- "daily"
    # numCores <- detectCores()
    # cl <- makePSOCKcluster(numCores)
    # predictions[[variable]][[response_curve_type]][["modelled"]]<- parLapplyLB(cl,pheno_list, pred_helper, .response_function_prediction., parameters,weight_parameter, granularity)
    # stopCluster(cl)
    
    
  }

# }


