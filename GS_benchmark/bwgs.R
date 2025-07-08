
library(BWGS)
library(data.table)
library(dplyr)
library(tibble)

#### pars
args <- commandArgs(TRUE)
model <- args[1]
train_geno <- args[2]
train_pheno <- args[3]
valid_geno <- args[4]
valid_pheno <- args[5]
outfile <- args[6]

setwd(getwd())

#### 1. read datasets
refGD <- fread(train_geno, header = TRUE) %>% as_tibble() %>%  column_to_rownames(var = names(.)[1])  
refPhe <- read.table(train_pheno, header = TRUE) 
tstGD <- fread(valid_geno, header = TRUE) %>% as_tibble() %>%  column_to_rownames(var = names(.)[1])  
tstPhe <- read.table(valid_pheno, header = TRUE) 

assign("drp", as.numeric(refPhe[, 3]))
eval(substitute(names(x) <- as.character(refPhe[, 1]), list(x = as.symbol("drp"))))

start_time <- proc.time()
#### 2. GS
res <- bwgs.predict(
    geno_train = refGD,
    pheno_train = drp,
    geno_target = tstGD,
    FIXED_train = "NULL",
    FIXED_target = "NULL",
    MAXNA = 0.2,
    MAF = 0.05,
    geno.impute.method = "NULL",
    geno.reduct.method = "NULL",
    reduct.size = "NULL",
    pval = "NULL",
    r2 = "NULL",
    MAP = "NULL",
    predict.method = model
)
end_time <- proc.time()
runtime <- end_time - start_time
print(paste0('Running time: ', runtime[3][[1]], ' s'))

#### 3. result
dgv <- data.frame(res, "member" = rownames(res))
final <- merge(dgv, tstPhe, by = "member")
r <- cor(final$gpred, final[, 6]) / mean(sqrt(final[, 5]))
b <- cov(final$gpred, final[, 6]) / var(final$gpred)
r2 <- mean(final$CD)
print(r)
print(b)

write.table(final, file = outfile, sep = "\t", quote = FALSE, row.names = FALSE)

other <- data.frame('file' = outfile,  'cor' = r, 'r2' = r2, 'basis'= b, 'runtime' = runtime[3][[1]])
write.table(other, file = paste0(outfile, '_statistics'), sep = "\t", quote = FALSE, row.names = FALSE)
