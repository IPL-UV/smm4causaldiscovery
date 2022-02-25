

idx <- c(5, 8, 10, 12 ,15, 17, 22, 28)
data <- lapply(idx, function(i){
  print(i)
  read.table(paste0("data/tuebingen_benchmark/pair",  sprintf("%04d",i),".txt"),
             sep = " ")
})

sapply(data, ncol)

library(ggplot2)
library(reshape2)

D <- melt(data, id.vars=c("V1", "V2"))
pp <- ggplot(D) + geom_hex(aes(x = V1, y = V2 ), bins = 30) + 
  xlab("")+ylab("")+
  facet_wrap(vars(L1), scales = "free", nrow = 2) + theme_bw() + 
  theme(
    legend.position = "none",
    strip.background = element_blank(),
    strip.text.x = element_blank(),
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(),panel.grid = element_blank()
  ) 

pp
ggsave("images/tubingen.pdf", plot = pp, width = 3.5, 
       height = 2,units = "in", dpi = 5)
