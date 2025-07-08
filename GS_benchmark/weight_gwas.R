# pars
args <- commandArgs(trailingOnly = TRUE)
gwas_file <- args[1]
bayes_file <- args[2]
output <- args[3]
PI <- 0.001
Navg <- 5


# load data
dat <- read.table(gwas_file, header=T)
# calculate LR
lr <- 0.5 * (dat$b / dat$se)^2
# calculate PI and Navg
Nleft <- round((Navg - 1) / 2)
if(Nleft < 0) Nleft <- 0
cat("SNP number=", length(lr), "\n")
cat("PI=", PI, "\n")
cat("Nleft=", Nleft, "\n")

avg <- matrix(0, nrow=length(lr), ncol=2)
for(i in 1:length(lr)) {
  avg[i, 2] <- mean(lr[max(1, i-Nleft):min(i+Nleft, length(lr))])
  avg[i, 1] <- PI * exp(avg[i, 2]) / (PI * exp(avg[i, 2]) + 1 - PI)
}
avg <- as.data.frame(avg)
avg$SNP <- dat$SNP
avg$p <- dat$p
avg$maf <- dat$Freq
colnames(avg)[1:2] <- c('weight', 'LR')
avg$weight <- sqrt(avg$weight)
write.table(avg[, c(3,1)], file=paste0(output,'_gwas_weight'), row.names=F, quote=F, sep="\t")



effect <- read.table(bayes_file, header=T)
effect$SNP <- sapply(strsplit(effect$SNP, "_"), function(x) {
    if (length(x) > 1) {
        paste(x[-length(x)], collapse = "_") 
    } else {
        x  
    }
})

ss = as.data.frame(avg[, c(3,1,4,5)])
dd = as.data.frame(effect[, c(1,2)])
if (all(row.names(ss) == row.names(dd))){
    res = cbind(ss, dd)
}

write.table(res[, c(1:4,6)], file=paste0(output,'_all_weight'), row.names=F, quote=F, sep="\t")


