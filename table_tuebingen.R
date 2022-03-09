library("reshape2")
library("xtable")

all_res <- melt(lapply(0:19, function(i){
  read.csv(paste0("results/tuebingen/rep", i, ".csv"))
}))

avg <- aggregate(value ~ X + variable, data = all_res, 
                 FUN = mean)
avg$X[avg$X == "smm_ensemble"] <- "CEMM"

std <- aggregate(value ~ X + variable, data = all_res, 
                 FUN = sd)
std$X[std$X == "smm_ensemble"] <- "CEMM"



tab.avg <- acast(avg,  X ~ variable)
tab.std <- acast(std,  X ~ variable)
tab.all <- matrix(paste0(format(tab.avg, digits = 2), " (",format(tab.std, digits = 1, ),")"), 
                  nrow = nrow(tab.avg), 
                  dimnames = list(rownames(tab.avg), colnames(tab.avg)))
select <- c("CEMM",  "rcc", "jarfo", "best", "vot", "avg", "fastANM", "RECI", "IGCI", "fastBV", "CDS")
idx <- which(rownames(tab.all) %in% select)
print(xtable(tab.all[select,, drop =FALSE]), booktabs = TRUE, include.rownames = TRUE)


