# DaFID-512
[danbooru-pretrained](https://github.com/RF5/danbooru-pretrained)を用いたFID計算を行います。<br>
ResNetを用いているのでFIDという名前は不適当ですが、わかりやすさのためにそうしています。<br>
計算部分のソースコードは一部[pytorch-fid](https://github.com/mseitzer/pytorch-fid)を借用しています。<br>

# Usage
`python3 danbooru-fid.py --A dirA --B dirB --batch-size 64`

通常のFIDと同じく、画像枚数は少なくとも10000枚以上必要です。
