# Date: 26/02/2021
# Author: Matthew Skiffington

# Nafis Sadat - sadatnfs (GitHub gist)
# https://gist.github.com/sadatnfs/8e73d23e375f361ecab845c5df8c488f

require(ncdf4)  
require(ncdf4.helpers)
require(data.table)
require(dplyr)

## Get the name of the value vars in the nc file
get_nc_value_name <- function(nc_file) {
  
  ## Get names
  nc_obj <- nc_open(nc_file)
  name<-names(nc_obj$var)
  
  ## Close file
  nc_close(nc_obj)
  
  ## Return the name
  return(name)
  
}

## Once we have the name of the variable we want to extract, we pass it onto this function to return the full dataset
xarray_nc_to_R <- function(nc_file, dimname, start=NA, count=NA, df_return = T) {
  
  ## Open the file and show the attribuets
  ncin <- nc_open(nc_file)
  print(ncin)
  
  ## Get the full array, using the variable name we want
  Rarray <- ncvar_get(ncin, dimname, start = start, count = count, collapse_degen=F)
  
  ## Get the fillvalue info
  fillvalue <- ncatt_get(ncin,dimname,"_FillValue")
  
  ## Get the dimension names in the right order
  array_dim <- ncdf4.helpers::nc.get.dim.names(ncin, dimname)
  
  
  ## Close the file
  nc_close(ncin)
  
  ## Get all of the dimension information in the order specified
  array_dim_list <- list()
  for(i in array_dim) {
    array_dim_list[[i]] <- ncin$dim[[i]]$vals
  }
  
  ## Fill in NaNs with NA
  Rarray[Rarray==fillvalue$value] <- NA
  
  
  ## Assign the dimension labels to the R array
  for(i in 1:length(array_dim_list)) {
    dimnames(Rarray)[[i]] <- array_dim_list[[i]]  
  }
  
  ## Attach the dimension name to the array
  names(attributes(Rarray)$dimnames) <- array_dim
  
  if(df_return) {
    return(data.frame(reshape2::melt(Rarray)))  
  } else {
    return(Rarray)  
  }
  
  
  
}

user <- 'skiff'
var1 <- 'RAIN_BC'
var2 <- 'SOILM'

xarray_data_rain <- sprintf("C:\\Users\\%s\\Desktop\\data\\VCSN\\ORIG\\MONTHLY\\%s\\VCSN_gridded_Rain_bc_1979-01_2019-12.nc",user,var1)
xarray_data_soilm <- sprintf("C:\\Users\\%s\\Desktop\\data\\VCSN\\ORIG\\MONTHLY\\%s\\VCSN_gridded_SoilM_1979-01_2019-12.nc",user,var2)

## We'll extract the 'rain' column and convert to a DT
## NOTE: We are pulling in the full xarray. If you wanted to slice, then you'd have
## to define the start and count variables... TBV

### Testing with VCSN MONTHLY rain data
dimname_rain <- get_nc_value_name(xarray_data_rain)[1]

vcsn_data_rain <- data.table(xarray_nc_to_R(nc_file = xarray_data_rain, dimname_rain))
print(vcsn_data_rain)

# Filter out NaN values
vcsn_data_rain_filtered <- vcsn_data_rain[!is.nan(vcsn_data_rain$value),] %>% rename(rain = value)

# VCSN MONTHLY SOILM data
dimname_soilm <- get_nc_value_name(xarray_data_soilm)[1]
vcsn_data_soilm <- data.table(xarray_nc_to_R(nc_file = xarray_data_soilm, dimname_soilm))
vcsn_data_soilm_filtered <- vcsn_data_soilm[!is.nan(vcsn_data_soilm$value),] %>% rename(soilm = value)

# length(unique(vcsn_data__rain_filtered$time))
# length(unique(vcsn_data__rain_filtered$lat))
# length(unique(vcsn_data__rain_filtered$lon))

combined_rain_soilm <- cbind(vcsn_data_soilm_filtered,vcsn_data_rain_filtered[,"rain"])
write.csv(x = combined_rain_soilm,file = "vcsn_rain_solim.csv", quote = FALSE)

# number of grid points
nrow(unique(vcsn_data_soilm_filtered[,c('lat','lon')]))

# number of unique cells (time and space)
nrow(unique(vcsn_data_soilm_filtered[,c('lat','lon')]))