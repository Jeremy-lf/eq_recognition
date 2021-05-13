from eq_model import EquationRecognition
# import matplotlib.pyplot as plt
import glob
import cv2
import os
import sys

if __name__ == '__main__':
    argv = sys.argv
    argc = len(argv)
    if argc > 2 : print("usage:\n%s image_directory", argv[0])
    else:
        image_dir = './test_data/'#argv[1]
        ER = EquationRecognition(gpu=True, device_id=0)
        image_list = os.listdir(image_dir)
        num = len(image_list)
        for idx, name in enumerate(image_list):
            if not(name.endswith(".bmp") or name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".png")):
                continue
            print("[%d/%d] %s" % (idx, num, name))

            try: 
                base_name, _ = os.path.splitext(name)
                img = cv2.imread(os.path.join(image_dir, name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                pred = ER.proc(img)

                print("%s -> %s" % (name, pred))
                with open(os.path.join(image_dir, base_name + '.txt'), 'w') as fw:
                    fw.write(pred)

                # plt.text(0, 0.6, r"$%s$" % pred, fontsize=50)

                # # hide axes
                # fig = plt.gca()
                # fig.axes.get_xaxis().set_visible(False)
                # fig.axes.get_yaxis().set_visible(False)
                # plt.draw()  # or savefig
                # plt.show()
            except:
                print("error")
                continue
