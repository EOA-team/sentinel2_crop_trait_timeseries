# Author: Flavian Tschurr
# Project: KP030
# Date: 10.05.2022
# Purpose: LAI dose response: Prepare input data of multiple fields
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
base_path_data <- "../../data/dose_response_in-situ"
granularity = "daily"
granularity = "hourly"
granularities <- c("hourly","daily")

################################################################################

aggregate2day <- function(time_vect, data_vect, aggregator){
  #'@param time_vect vector with timestamps in it
  #'@param data_vect vector with data which will be aggregated
  #'@param aggregator method which will be applied to aggregate (mean, min, max, sum)
  #'@return vector with daily values
  
  require(lubridate)
  days <- as.Date(time_vect)
  # time_vect <- as.POSIXct(time_vect)
  data_out <- NULL
  day_counter <- 1
  
  for(d in unique(days)){
    if(aggregator=="mean"){
      data_out[day_counter] <- mean(as.numeric(data_vect[which(days== d)]),na.rm=TRUE)
      
    } else if( aggregator == "min"){
      data_out[day_counter] <- min(as.numeric( data_vect[which(days== d)]),na.rm=TRUE)
      
    }else if(aggregator=="max"){
      data_out[day_counter] <- max(as.numeric( data_vect[which(days== d)]),na.rm=TRUE)
      
    } else if(aggregator == "sum"){
      data_out[day_counter] <- sum(as.numeric( data_vect[which(days== d)]),na.rm=TRUE)
      
    }
    day_counter <- day_counter+1
  } # end day loop
  
  return(data_out)
}

for(granularity in granularities){
  
  
  ################################################################################
  ################################################################################
  # bramenwies
  ################################################################################
  ################################################################################
  ################################################################################
  location_id = "CH_Bramenwies"
  
  
  ################################################################################
  # read LAI data
  ################################################################################
  LAI <- read.csv(paste(base_path_data,location_id,"LAI_BW_Raw-Data.csv",sep="/"))
  LAI$Point_ID <- as.factor(paste0("p_",LAI$Point_ID))
  
  LAI$timestamp <- as.POSIXct(LAI$dateonly)
  
  df_LAI <- LAI %>%
    mutate(DOY = yday(timestamp)) %>%
    group_by(Point_ID) %>%
    arrange(DOY) %>%
    mutate(LAI_smooth = smooth.spline(x= DOY, y = LAI_value)$y)%>%
    mutate(lag_DOY = lag(DOY),
           lag_LAI = lag(LAI_value),
           delta_LAI = (LAI_value - lag_LAI),
           lag_LAI_smooth = lag(LAI_smooth),
           delta_LAI_smooth = (LAI_smooth - lag_LAI_smooth))
  
  df_LAI <- subset(df_LAI, Point_ID != "p_24")
  df_LAI <- subset(df_LAI, Point_ID != "p_3")
  df_LAI <- subset(df_LAI, Point_ID != "p_20")
  
  
  library(ggplot2)
  
  ggplot(data=df_LAI, aes(timestamp,LAI_smooth, color=Point_ID))+
    geom_line()+ theme_classic()
  ################################################################################
  # read env data
  ################################################################################
  env_variable = "T_mean"
  
  env <- read.csv(paste(base_path_data,location_id,"Meteo_BW.csv",sep="/"))
  env$timestamp <- as.POSIXct(env$time)
  env <- env %>%
    arrange(timestamp)
  
  if(granularity == "daily"){
    
    dates <- unique(as.Date(env$timestamp))
    tas <- aggregate2day(time_vect = env$timestamp,data_vect = env[[env_variable]], aggregator = "mean")
    env <- data.frame(timestamp = dates, "T_mean"= tas)
    
  }
  
  ################################################################################
  # clean and combine data
  ################################################################################
  
  variable <-"delta_LAI_smooth"
  measurement_list <- list()
    
    all_points <- list()
    for(pl in unique(df_LAI$Point_ID)) {
      all_points[[pl]] <- df_LAI %>%
        filter(Point_ID == pl) #%>%
    }
    
    measurement_list <-lapply(all_points, 
                              combined_data_cleaning_function,
                              df_env = env,
                              variable=variable,
                              env_variable = env_variable,
                              data_cleaning="no_negative_values")
    
    combined_measurement_list <- list()
    for(point_ in names(measurement_list)){
      for(measurement in names(measurement_list[[point_]])){
        combined_measurement_list[[measurement]]<-  measurement_list[[point_]][[measurement]]
      }
    }
    
  
  
  
  
  ################################################################################
  ################################################################################
  ################################################################################
  #MNI
  ################################################################################
  ################################################################################
  ################################################################################
  location_id = "DE_MNI"
    
  MNI_files <- list.files(paste(base_path_data,location_id,sep="/"))
  
  LAI_files <- MNI_files[grep("LAI_",MNI_files)]
  LAI_files <- LAI_files[grep(".csv",LAI_files)]
  
  LAI_MNI <- list()
  for(f in LAI_files){
    df_LAI <- read.csv(paste(base_path_data,location_id,f,sep="/"))  
    df_LAI$timestamp <- as.Date(df_LAI$date)
    
    df_LAI <- df_LAI %>%
      mutate(DOY = yday(timestamp)) %>%
      arrange(DOY) %>%
      mutate(LAI_smooth = smooth.spline(x= DOY, y = LAI_value)$y)%>%
      mutate(lag_DOY = lag(DOY),
             lag_LAI = lag(LAI_value),
             delta_LAI = (LAI_value - lag_LAI),
             lag_LAI_smooth = lag(LAI_smooth),
             delta_LAI_smooth = (LAI_smooth - lag_LAI_smooth))
    
    
    LAI_MNI[[f]] <- subset(df_LAI, BBCH >=30 & BBCH <=61)
    
  }
  
  LAI_MNI_df <- do.call("rbind", LAI_MNI)
  
  
  Meteo_files <- MNI_files[grep("Meteo_",MNI_files)]
  
  Meteo_MNI <- list()
  for(f in Meteo_files){
    Meteo_MNI[[f]] <- read.csv(paste(base_path_data,location_id,f,sep="/"))  
  }
  Meteo_MNI_df <- do.call("rbind", Meteo_MNI)
  Meteo_MNI_df$timestamp <- as.POSIXct(Meteo_MNI_df$time)
  
  if(granularity == "daily"){
    
    dates <- unique(as.Date(Meteo_MNI_df$timestamp))
    tas <- aggregate2day(time_vect = Meteo_MNI_df$timestamp,data_vect = Meteo_MNI_df[[env_variable]], aggregator = "mean")
    Meteo_MNI_df <- data.frame(timestamp = dates, "T_mean"= tas)
    
  }
  
  variable <-"delta_LAI_smooth"
  env_variable = "T_mean"
  # MNI_measurement_list <- combined_data_cleaning_function(LAI_MNI_df, Meteo_MNI_df, variable,env_variable,data_cleaning="no_negative_values")
  # 
  
  MNI_measurement_list_long <-lapply(LAI_MNI, 
                            combined_data_cleaning_function,
                            df_env = Meteo_MNI_df,
                            variable=variable,
                            env_variable = env_variable,
                            data_cleaning="no_negative_values")
  
  MNI_measurement_list <- list()
  for(point_ in names(MNI_measurement_list_long)){
    for(measurement in names(MNI_measurement_list_long[[point_]])){
      MNI_measurement_list[[measurement]]<-  MNI_measurement_list_long[[point_]][[measurement]]
    }
  }
  
  
  ################################################################################
  ################################################################################
  ################################################################################
  #Rur
  ################################################################################
  ################################################################################
  ################################################################################
  location_id = "DE_Rur"
  
  Rur_files <- list.files(paste(base_path_data,location_id,sep="/"))
  
  LAI_files <- Rur_files[grep("LAI_",Rur_files)]
  LAI_files <- LAI_files[grep(".csv",LAI_files)]
  
  LAI_Rur <- list()
  for(f in LAI_files){
    df_LAI <- read.csv(paste(base_path_data,location_id,f,sep="/"))  
    if(length(which(names(df_LAI) == "BBCH"))==0){
      print("XX")
      # browser()
      next
    }
    print(f)
     df_LAI$timestamp <- as.Date(df_LAI$date)
     df_LAI <- df_LAI[which(!is.na(df_LAI$LAI_value)),]
    if(length(df_LAI$LAI_value)<=4){
      df_LAI$LAI_smooth = df_LAI$LAI_value
      
      df_LAI <- df_LAI %>%
        mutate(DOY = yday(timestamp)) %>%
        arrange(DOY) %>%
        mutate(lag_DOY = lag(DOY),
               lag_LAI = lag(LAI_value),
               delta_LAI = (LAI_value - lag_LAI),
               lag_LAI_smooth = lag(LAI_smooth),
               delta_LAI_smooth = (LAI_smooth - lag_LAI_smooth))
      
    }else{
      
      df_LAI <- df_LAI %>%
        mutate(DOY = yday(timestamp)) %>%
        arrange(DOY) %>%
        mutate(LAI_smooth = smooth.spline(x= DOY, y = LAI_value)$y)%>%
        mutate(lag_DOY = lag(DOY),
               lag_LAI = lag(LAI_value),
               delta_LAI = (LAI_value - lag_LAI),
               lag_LAI_smooth = lag(LAI_smooth),
               delta_LAI_smooth = (LAI_smooth - lag_LAI_smooth))
      
    }
    
    if(dim(subset(df_LAI, BBCH >=30 & BBCH <=61))[1] <= 1){
      next
    }else{
      
      LAI_Rur[[f]] <- subset(df_LAI, BBCH >=30 & BBCH <=61)
    }
    
  }
  
  LAI_Rur_df <- do.call("rbind", LAI_Rur)
  
  
  Meteo_files <- Rur_files[grep("Meteo_",Rur_files)]
  
  Meteo_Rur <- list()
  for(f in Meteo_files){
    Meteo_Rur[[f]] <- read.csv(paste(base_path_data,location_id,f,sep="/"))  
  }
  Meteo_Rur_df <- do.call("rbind", Meteo_Rur)
  Meteo_Rur_df$timestamp <- as.POSIXct(Meteo_Rur_df$time)
  Meteo_Rur_df <- Meteo_Rur_df[order(Meteo_Rur_df$timestamp),]
  
  if(granularity == "daily"){
    
    dates <- unique(as.Date(Meteo_Rur_df$timestamp))
    tas <- aggregate2day(time_vect = Meteo_Rur_df$timestamp,data_vect = Meteo_Rur_df[[env_variable]], aggregator = "mean")
    Meteo_Rur_df <- data.frame(timestamp = dates, "T_mean"= tas)
    
  }
  
  variable <-"delta_LAI_smooth"
  env_variable = "T_mean"
  # Rur_measurement_list <- combined_data_cleaning_function(LAI_Rur_df, Meteo_Rur_df, variable,env_variable,data_cleaning="no_negative_values")
  
  Rur_measurement_list_long <-lapply(LAI_Rur, 
                                     combined_data_cleaning_function,
                                     df_env = Meteo_Rur_df,
                                     variable=variable,
                                     env_variable = env_variable,
                                     data_cleaning="no_negative_values")
  
  Rur_measurement_list <- list()
  for(point_ in names(Rur_measurement_list_long)){
    for(measurement in names(Rur_measurement_list_long[[point_]])){
      Rur_measurement_list[[measurement]]<-  Rur_measurement_list_long[[point_]][[measurement]]
    }
  }
  
  
  
  #################################################################################
  
  all_locations <- list()
  
  for(i in names(combined_measurement_list)){
    all_locations[[i]] <- combined_measurement_list[[i]]
  }
  
  for(i in names(MNI_measurement_list)){
    all_locations[[i]] <- MNI_measurement_list[[i]]
  }
  
  for(i in names(Rur_measurement_list)){
    all_locations[[i]] <- Rur_measurement_list[[i]]
  }
  
  
  saveRDS(all_locations, paste0("../../data/dose_reponse_in-situ/output/LAI_",granularity,"_Bramenwies_MNI_Rur.rds"))
  
}
