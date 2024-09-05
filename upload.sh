#!/usr/bin/bash


/usr/bin/tar -cvf ipc-$(date "+%F").tar /home/zum/monitor/vid_storage/* &&
/usr/bin/tar -cvf cam-$(date "+%F").tar /home/zum/cam/rec/* &&
/usr/bin/tar -cvf ipc-$(date "+%F").tar /home/zum/sentry/9D0B0EDPAGCE34C/* &&
/usr/bin/sleep 1s &&
/usr/bin/gsutil cp *.tar gs://zumstation_sentry_storage0/ &&
touch all_done && echo "successfully uploaded" > all_done
