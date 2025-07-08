library(hibayes)
library(data.table)
library(dplyr)
library(tibble)

setwd(getwd())

#### pars
args <- commandArgs(TRUE)
model <- args[1]
train_geno <- args[2]
train_pheno <- args[3]
valid_geno <- args[4]
valid_pheno <- args[5]
mapfile <- args[6]
outfile <- args[7]

niter <- 20000
nburn <- 12000
thin <- 5
nthread <- 20


#### 1. read datasets
refGD <- fread(train_geno, header = TRUE) %>% as_tibble() %>%  column_to_rownames(var = names(.)[1])  
refPhe <- read.table(train_pheno, header = TRUE, row.names=1) 
tstGD <- fread(valid_geno, header = TRUE) %>% as_tibble() %>%  column_to_rownames(var = names(.)[1])  
tstPhe <- read.table(valid_pheno, header = TRUE, row.names=1) 
map <- read.table(mapfile, header = F)[, c(2,1,4)]

#### 2. sort
matched_order <- match(row.names(refGD), row.names(refPhe))
refGD <- refGD[matched_order, ]
refPhe <- refPhe[matched_order, 2,drop=F]

matched_order <- match(row.names(tstGD), row.names(tstPhe))
tstGD <- tstGD[matched_order, ]
tstPhe <- tstPhe[matched_order, 2,drop=F]
tstPhe[,1] <- NA

#### 3. merge
colnames(tstGD) <- colnames(refGD) 
GD <- rbind(refGD, tstGD)
pheno <- rbind(refPhe, tstPhe)
pheno <- data.frame('member' = row.names(pheno), 'drp' = pheno[,1])
row.names(pheno) <- pheno[,1]
identical(rownames(pheno), rownames(GD))
geno.id <- row.names(GD)


#### bayes models
start_time <- proc.time()
if (model %in% c("BayesBpi", "BayesCpi")) {
    fitCpi <- bayes(drp ~ 1,
        data = pheno, M = GD, printfreq = 100,
        M.id = geno.id, method = model, Pi = c(0.98, 0.02),
        niter = niter, nburn = nburn, thin = thin,
        seed = 666666, verbose = TRUE, threads = nthread
    )
} else if (model == "BayesR") {
    fitCpi <- bayes(drp ~ 1,
        data = pheno, M = GD, printfreq = 100,
        M.id = geno.id, method = "BayesR", niter = niter,
        nburn = nburn, thin=thin, Pi = c(0.95, 0.02, 0.02, 0.01),
        fold = c(0, 0.0001, 0.001, 0.01), seed = 666666, map = map, windsize = 1e6, threads = nthread
    )
} else if (model == "BayesA") {
    fitCpi <- bayes(drp ~ 1,
        data = pheno, M = GD, M.id = geno.id, method = model,
        printfreq = 100, niter = niter, nburn = nburn, thin = thin,
        seed = 666666, verbose = TRUE, threads = nthread
    )
}

#### 结束时间
end_time <- proc.time()
runtime <- end_time - start_time
print(paste0('Running time: ', runtime[3][[1]], ' s'))

save(fitCpi, file = paste0(outfile, ".RData"))

#### result
res <- fitCpi[["g"]]
names(res) <- c("member", "gpred")

tstPhe <- read.table(valid_pheno, header = TRUE) 
final <- merge(res, tstPhe, by = "member") ### 验证群体的预测结果
r <- cor(final$gpred, final[, 4]) / mean(sqrt(final[, 3]))
b <- cov(final$gpred, final[, 4]) / var(final$gpred)
print(r)
print(b)

#### save
write.table(final, file = outfile, sep = "\t", quote = FALSE, row.names = FALSE)

other <- data.frame('file' = outfile,  'cor' = r, 'basis'= b, 'runtime' = runtime[3][[1]], 'h2' = fitCpi[["h2"]])
write.table(other, file = paste0(outfile, '_statistics'), sep = "\t", quote = FALSE, row.names = FALSE)

pve <- apply(as.matrix(refGD), 2, var) * (fitCpi[["alpha"]]^2) / var(refPhe[, 1])
snp_effects <- data.frame('SNP' = colnames(GD), 'effect' = fitCpi[["alpha"]], 'pve' = pve)
write.table(snp_effects, paste0(outfile, '_SNP_effect'), sep = "\t", quote = FALSE, row.names=F)


