import cv2
import requests
from picamera2 import MappedArray, Picamera2, Preview
from pyzbar import pyzbar
from pyzbar.pyzbar import decode

colour = (0, 255, 0)
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
thickness = 2
def draw_barcodes(request):
    with MappedArray(request, "main") as m:
        for b in barcodes:
            if b.polygon:
                x = min([p.x for p in b.polygon])
                y = min([p.y for p in b.polygon]) - 30
                cv2.putText(m.array, b.data.decode('utf-8'), (x, y), font, scale, colour, thickness)
picam2 = Picamera2()
picam2.start_preview(Preview.NULL) #or QT as required).
config = picam2.create_preview_configuration(main={"size": (1280, 960)})
picam2.configure(config)
barcodes = []
picam2.post_callback = draw_barcodes
picam2.start()
def getserial():
    # Extract serial from cpuinfo file
    cpuserial = "0000000000000000"
    try:
        f = open('/proc/cpuinfo', 'r')
        for line in f:
            if line[0:6] == 'Serial':
                cpuserial = line[10:26]
        f.close()
    except:
        cpuserial = "ERROR000000000"

    return cpuserial
while True:
    rgb = picam2.capture_array("main")
    frame = picam2.capture_array()
    barcodes = decode(rgb)
    qrcodes = pyzbar.decode(frame)
    for barcode in qrcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        params = {
            'auth': 'VqIJQvhPVPQOGUvDzBgSHvNeMAojle',
            'qrcode': barcodeData,
            'serialNumber': getserial()
        }
        x = requests.get("https://zy.zumrides.com/api/api_get_qr_cycle_stop.php", params=params)
        print(x)
        text = "{} ({})".format(barcodeData, barcodeType)
        print(text)
        #cv2.imshow("piCam", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
