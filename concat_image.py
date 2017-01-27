import os
import argparse
from PIL import Image
from PIL import ImageDraw
import glob

parser = argparse.ArgumentParser(
    description='concat image')
parser.add_argument('--filters', '-f', type=int, default=-1,
                    help='Number of filters')
parser.add_argument('--top', '-t', type=int, default=10,
                    help='gather top n images')
parser.add_argument('--cols', type=int, default=5,
                    help='columns')
parser.add_argument('--scale', '-s', type=int, default=1,
                    help='filter scale')
parser.add_argument('--pad', '-p', type=int, default=1,
                    help='filter padding')
parser.add_argument('--quality', '-q', type=int, default=100,
                    help='filter padding')
parser.add_argument('--mean', '-m', nargs=3, default=[124, 117, 104],
                    help='Output directory')
parser.add_argument('--bordercolor', '-b', nargs=3, default=[255, 255, 255],
                    help='Output directory')
parser.add_argument('--outdirprefix', '-pf', default='concat_',
                    help='Output directory')
parser.add_argument('--out', '-o', default=None,
                    help='Output directory')
parser.add_argument('--root', '-R', default='.',
                    help='Root directory path of image files')
#parser.add_argument('--test', action='store_true')
#parser.set_defaults(test=False)

if __name__ == '__main__':
    args = parser.parse_args()
    max_size = 0
    pad = args.pad
    top = args.top
    cols = args.cols
    rows = (top + cols - 1) // cols
    scale = args.scale
    bgcolor = tuple(args.mean)
    bcolor = tuple(args.bordercolor)

    outdir = args.out or os.path.join(
        os.path.dirname(args.root),
        args.outdirprefix + os.path.basename(args.root))

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if args.filters < 0:
        files = sorted(glob.glob(os.path.join(args.root, '*[0-9]_00.png')))
        last_num = os.path.basename(files[len(files) - 1])[:4]
        args.filters = int(last_num) + 1

    for f in range(args.filters):
        images = []
        for k in range(top):
            filename = os.path.join(args.root,
                "{0:0>4}_{1:0>2}.png".format(f, k))
            im = Image.open(filename)
            im = im.resize((im.size[0] * scale, im.size[1] * scale))
            images.append(im)
            max_size = max(im.size[0], im.size[1], max_size)

        step = max_size + pad
        W = step * cols + pad
        H = step * rows + pad
        tiled_image = Image.new("RGB", (W, H))
        draw = ImageDraw.Draw(tiled_image)
        draw.rectangle(((0,0),(W,H)), bgcolor)
        # if pad > 0, draw border
        if pad > 0:
            for r in range(rows + 1):
                draw.rectangle(((0, r * step),(W, r * step + pad - 1)), fill=bcolor)
            for c in range(cols + 1):
                draw.rectangle(((c * step, 0),(c * step + pad -1, H)), fill=bcolor)
        for k, im in enumerate(images):
            r, c = k // cols, int(k % cols)
            x, y = c * step + pad, r * step + pad
            tiled_image.paste(im, (x, y))

        print(f, max_size)
        tiled_image.save(os.path.join(outdir, '{0:0>4}.jpg'.format(f)),
            'JPEG', quality=args.quality, optimize=True)
