#!/bin/bash

gen=$1
phe=$2
tst_gen=$3
tst_phe=$4
bayesBpi_effect=$5
nthread=$6
outfile=$7


#### ====================================
#### GWAS
#### ====================================
start=`date +%s`
#### 4.1 build matrix
gcta64 --bfile $gen --autosome-num 29 --make-grm --out $outfile --thread-num $nthread
#### 4.2 calculate pca
gcta64 --grm $outfile --pca 10 --thread-num $nthread --out $outfile

awk '{print $1"\t"$1"\t"$3}' $phe > $outfile.drp
#### 4.3 calculate h2
gcta64 --grm $outfile --pheno $outfile.drp --reml --out $outfile --thread-num $nthread
#### 4.4 run gwas
gcta64 --mlma --bfile $gen --grm $outfile --pheno $outfile.drp --pca 5 --out $outfile --thread-num $nthread
end=`date +%s`
echo "GWAS running time: "$((end-start))"s"

#### 4.5 calculate SNP weight
Rscript ./02models/WGBLUP/weight_gwas.R $outfile.mlma ${bayesBpi_effect} $outfile
Rscript ./02models/WGBLUP/weight_bayesBpi.R $outfile.mlma ${bayesBpi_effect} $outfile
# #### merge train and tst
plink --bfile $gen --bmerge ${tst_gen}.bed ${tst_gen}.bim ${tst_gen}.fam --cow --make-bed --out ${outfile}_all


#### ====================================
#### WGBLUP
#### ====================================
WGBLUP(){
    gen=$1
    phe=$2
    weight=$3
    out=$4
    nthread=$5
    start=`date +%s`
    # generate weighted GRM, BayesB
    hiblup --make-xrm \
        --code-method 1 \
        --bfile $gen \
        --add \
        --snp-weight $weight \
        --thread $nthread \
        --out $out
    ## run WBLUP, BayesB
    hiblup --single-trait \
        --pheno $phe \
        --pheno-pos 3 \
        --xrm $out.GA \
        --add \
        --r2 \
        --threads $nthread \
        --out ${out}.wgblup_gebv
    end=`date +%s`
    echo $((end-start))
}

wgblup_gwas=$(WGBLUP ${outfile}_all $phe ${outfile}_gwas_weight $outfile $nthread)
wgblup_bayesBpi=$(WGBLUP ${outfile}_all $phe ${outfile}_BayesBpi_weight $outfile $nthread)

#### 4.7 GBLUP
start=`date +%s`
# generate GRM
hiblup --make-xrm \
    --code-method 1 \
    --bfile ${outfile}_all \
    --add \
    --thread $nthread \
    --out $outfile.gblup

## run GBLUP
hiblup --single-trait \
    --pheno $phe \
    --pheno-pos 3 \
    --xrm $outfile.gblup.GA \
    --add \
    --r2 \
    --threads $nthread \
    --out ${outfile}.gblup_gebv
end=`date +%s`
echo "WGBLUP_gblup running time: "$((end-start))"s"

#### calculate r2
Rscript /public/home/2018013/02users/zwj/02master/GS_article/02models/WGBLUP/WGBLUP_cor.R \
    ${outfile}.gblup_gebv.rand  ${outfile}.wgblup_gebv.rand ${tst_phe} ${outfile}
