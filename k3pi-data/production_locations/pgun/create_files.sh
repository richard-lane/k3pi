#!/bin/bash
# Create production location text files
# You don't need to run this, the text files are already created
# This file just exists so you can see how

set -e

PROD_DIR='/eos/lhcb/user/n/njurik/D02Kpipipi/PGun/Tuples'
DUMP_DIR='k3pi-data/production_locations/pgun'
CF_CODE=27165071
DCS_CODE=27165072

echo '[ ################ ]'
echo -n '  '

# Assumed that the ones not in a labelled dir (e.g. 2018_dw/) are 2018 magup
# maxdepth option so that we dont go into the 2018_dw/ etc dirs
# 2018 magup
find $PROD_DIR/$CF_CODE/ -maxdepth 2 -type f -name 'Hlt1TrackMVA*' > $DUMP_DIR/cf_2018_magup/Hlt1TrackMVA.txt
echo -n .
find $PROD_DIR/$CF_CODE/ -maxdepth 2 -type f -name 'pGun_TRACK*' > $DUMP_DIR/cf_2018_magup/pGun_TRACK.txt
echo -n .

find $PROD_DIR/$DCS_CODE/ -maxdepth 2 -type f -name 'Hlt1TrackMVA*' > $DUMP_DIR/dcs_2018_magup/Hlt1TrackMVA.txt
echo -n .
find $PROD_DIR/$DCS_CODE/ -maxdepth 2 -type f -name 'pGun_TRACK*' > $DUMP_DIR/dcs_2018_magup/pGun_TRACK.txt
echo -n .

# 2018 magdown
find $PROD_DIR/$CF_CODE/2018_dw/ -type f -name 'Hlt1TrackMVA*' > $DUMP_DIR/cf_2018_magdown/Hlt1TrackMVA.txt
echo -n .
find $PROD_DIR/$CF_CODE/2018_dw/ -type f -name 'pGun_TRACK*' > $DUMP_DIR/cf_2018_magdown/pGun_TRACK.txt
echo -n .

find $PROD_DIR/$DCS_CODE/2018_dw/ -type f -name 'Hlt1TrackMVA*' > $DUMP_DIR/dcs_2018_magdown/Hlt1TrackMVA.txt
echo -n .
find $PROD_DIR/$DCS_CODE/2018_dw/ -type f -name 'pGun_TRACK*' > $DUMP_DIR/dcs_2018_magdown/pGun_TRACK.txt
echo -n .

# 2016 magdown
find $PROD_DIR/$CF_CODE/2016_dw/ -type f -name 'Hlt1TrackMVA*' > $DUMP_DIR/cf_2016_magdown/Hlt1TrackMVA.txt
echo -n .
find $PROD_DIR/$CF_CODE/2016_dw/ -type f -name 'pGun_TRACK*' > $DUMP_DIR/cf_2016_magdown/pGun_TRACK.txt
echo -n .

find $PROD_DIR/$DCS_CODE/2016_dw/ -type f -name 'Hlt1TrackMVA*' > $DUMP_DIR/dcs_2016_magdown/Hlt1TrackMVA.txt
echo -n .
find $PROD_DIR/$DCS_CODE/2016_dw/ -type f -name 'pGun_TRACK*' > $DUMP_DIR/dcs_2016_magdown/pGun_TRACK.txt
echo -n .

# 2016 magup
find $PROD_DIR/$CF_CODE/2016_up/ -type f -name 'Hlt1TrackMVA*' > $DUMP_DIR/cf_2016_magup/Hlt1TrackMVA.txt
echo -n .
find $PROD_DIR/$CF_CODE/2016_up/ -type f -name 'pGun_TRACK*' > $DUMP_DIR/cf_2016_magup/pGun_TRACK.txt
echo -n .

find $PROD_DIR/$DCS_CODE/2016_up/ -type f -name 'Hlt1TrackMVA*' > $DUMP_DIR/dcs_2016_magup/Hlt1TrackMVA.txt
echo -n .
find $PROD_DIR/$DCS_CODE/2016_up/ -type f -name 'pGun_TRACK*' > $DUMP_DIR/dcs_2016_magup/pGun_TRACK.txt
echo .

# Sort these files so that the files are in a consistent order, hopefully
# 18 up
echo -n '  '
sort $DUMP_DIR/cf_2018_magup/Hlt1TrackMVA.txt > tmp.txt
mv tmp.txt $DUMP_DIR/cf_2018_magup/Hlt1TrackMVA.txt
echo -n .
sort $DUMP_DIR/cf_2018_magup/pGun_TRACK.txt > tmp.txt
mv tmp.txt $DUMP_DIR/cf_2018_magup/pGun_TRACK.txt
echo -n .

sort $DUMP_DIR/dcs_2018_magup/Hlt1TrackMVA.txt > tmp.txt
mv tmp.txt $DUMP_DIR/dcs_2018_magup/Hlt1TrackMVA.txt
echo -n .
sort $DUMP_DIR/dcs_2018_magup/pGun_TRACK.txt > tmp.txt
mv tmp.txt $DUMP_DIR/dcs_2018_magup/pGun_TRACK.txt
echo -n .

# 18 down
sort $DUMP_DIR/cf_2018_magdown/Hlt1TrackMVA.txt > tmp.txt
mv tmp.txt $DUMP_DIR/cf_2018_magdown/Hlt1TrackMVA.txt
echo -n .
sort $DUMP_DIR/cf_2018_magdown/pGun_TRACK.txt > tmp.txt
mv tmp.txt $DUMP_DIR/cf_2018_magdown/pGun_TRACK.txt
echo -n .

sort $DUMP_DIR/dcs_2018_magdown/Hlt1TrackMVA.txt > tmp.txt
mv tmp.txt $DUMP_DIR/dcs_2018_magdown/Hlt1TrackMVA.txt
echo -n .
sort $DUMP_DIR/dcs_2018_magdown/pGun_TRACK.txt > tmp.txt
mv tmp.txt $DUMP_DIR/dcs_2018_magdown/pGun_TRACK.txt
echo -n .

# 16 down
sort $DUMP_DIR/cf_2016_magdown/Hlt1TrackMVA.txt > tmp.txt
mv tmp.txt $DUMP_DIR/cf_2016_magdown/Hlt1TrackMVA.txt
echo -n .
sort $DUMP_DIR/cf_2016_magdown/pGun_TRACK.txt > tmp.txt
mv tmp.txt $DUMP_DIR/cf_2016_magdown/pGun_TRACK.txt
echo -n .

sort $DUMP_DIR/dcs_2016_magdown/Hlt1TrackMVA.txt > tmp.txt
mv tmp.txt $DUMP_DIR/dcs_2016_magdown/Hlt1TrackMVA.txt
echo -n .
sort $DUMP_DIR/dcs_2016_magdown/pGun_TRACK.txt > tmp.txt
mv tmp.txt $DUMP_DIR/dcs_2016_magdown/pGun_TRACK.txt
echo -n .

# 16 up
sort $DUMP_DIR/cf_2016_magup/Hlt1TrackMVA.txt > tmp.txt
mv tmp.txt $DUMP_DIR/cf_2016_magup/Hlt1TrackMVA.txt
echo -n .
sort $DUMP_DIR/cf_2016_magup/pGun_TRACK.txt > tmp.txt
mv tmp.txt $DUMP_DIR/cf_2016_magup/pGun_TRACK.txt
echo -n .

sort $DUMP_DIR/dcs_2016_magup/Hlt1TrackMVA.txt > tmp.txt
mv tmp.txt $DUMP_DIR/dcs_2016_magup/Hlt1TrackMVA.txt
echo -n .
sort $DUMP_DIR/dcs_2016_magup/pGun_TRACK.txt > tmp.txt
mv tmp.txt $DUMP_DIR/dcs_2016_magup/pGun_TRACK.txt
echo .
