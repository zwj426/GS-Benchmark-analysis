# pars
args <- commandArgs(trailingOnly = TRUE)
gwas_file <- args[1]
bayes_file <- args[2]
output <- args[3]
PI <- 0.001
Navg <- 5


load(paste0(bayes_file, '.RData'))
# calculate SNP SE
nums <- nrow(fitCpi$MCMCsamples$alpha)
snp_se <- sapply(seq(1:nums), function(i) {
    sd(fitCpi$MCMCsamples$alpha[i,])
})

data <- read.table(paste0(bayes_file, '_SNP_effect'), header=T)
data$SNP <- sapply(strsplit(data$SNP, "_"), function(x) {
    if (length(x) > 1) {
        paste(x[-length(x)], collapse = "_")  
    } else {
        x  
    }
})

bayes_weight <- data.frame('SNP' = data$SNP, 'effect' = fitCpi[["alpha"]], se=snp_se)
PI <- fitCpi[['pi']][2]
lr <- 0.5 * (bayes_weight$effect / bayes_weight$se)^2
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
avg$SNP <- bayes_weight$SNP
colnames(avg)[1:2] <- c('weight', 'LR')
avg$BayesBpi_effect <- bayes_weight$effect
avg[, c('BayesBpi_effect', 'BayesBpi_se')] <- bayes_weight[, c('effect', 'se')]
avg$se <- sqrt(avg$weight)
write.table(avg[, c(3,1)], file=paste0(output,'_BayesBpi_weight'), row.names=F, quote=F, sep="\t")



gwas <- read.table(gwas_file, header=T)
all(gwas$SNP == avg$SNP)
ss = as.data.frame(avg[, c(3,1,2,4,5)])
dd = as.data.frame(gwas[, c(2,1,3,6,7,8,9)])
if (all(row.names(ss) == row.names(dd))){
    res = cbind(dd, ss)
}

write.table(res[, c(1:7,9:12)], file=paste0(output,'_all_weight'), row.names=F, quote=F, sep="\t")


