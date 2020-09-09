import cv2
import shutil
cam = cv2.VideoCapture(0)

cv2.namedWindow("input")

img_counter = 0
while True:
	ret, frame = cam.read()
	if not ret:
		print(failure)
		break
	cv2.imshow("input", frame)

	k = cv2.waitKey(1)
	if k%256 == 27:
		#ifpress esc close
		print("closing")
		break
	elif k%256 == 32:
		#if press space take pic
		img_name = "input{}.png"
		cv2.imwrite(img_name, frame)
		print("{} written!".format(img_name))
		img_counter += 1
		#so picture isnt just black
cam.release()

cv2.destroyAllWindows()

source="C:\\Users\\ozcon\\Desktop\\Desktop\\Python-CS\\input{}.png"
destination="C:\\Users\\ozcon\\Desktop\\ee"

new_path = shutil.move(source, destination)

print(new_path)