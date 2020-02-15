# https://mobiusstripblog.wordpress.com/2016/12/26/first-blog-post/

from lidar_lite import Lidar_Lite
lidar = Lidar_Lite()

connected = lidar.connect(1)

if connected < -1:
    print ("Not Connected")
else:
    print ("Connected")

for i in range(100):
    distance = lidar.getDistance()
    print("Distance to target = %s" % (distance))
    if int(distance) < 50:
        print("Too Close!!! Back Off!!!")