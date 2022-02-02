library("reshape2")
RN <- c("acc", "t.train", "t.test")
PRF <- c("", "", "_s", "_ntrain", "_ntest", "_gamma")

load_results <- function(mechs, ncoefs, sizes, 
                         ntrains, ntests, gammas, dir = "results", 
                         exp = "generated_data", nrep = 10){
  
  exp_combin <- expand.grid(mech = mechs, ncoef = ncoefs, 
                            size = sizes, ntrain = ntrains, 
                            ntest = ntests, gamma = gammas, 
                            stringsAsFactors = FALSE)
  
  res <- lapply(1:nrow(exp_combin), function(i) {
    basepath <- file.path(dir, exp, paste0(PRF, exp_combin[i,], collapse = "")) 
    print(basepath)
    files <- file.path(basepath, paste0("rep", 0:(nrep - 1), ".csv"))
    lapply(files, function(f){
      tmp <- t(read.csv(f, row.names = RN))
      cbind(tmp,
            exp_combin[i,],
            alg = row.names(tmp))})
  })
  melt(res, measure.vars = c("acc", "t.train", "t.test"))
}
