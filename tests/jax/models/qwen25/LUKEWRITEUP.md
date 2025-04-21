## start


lspci | grep -i tenstorrent 
didnt run

installed to check hardware details
apt-get update && apt-get install -y pciutils

lspci | grep -i tenstorrent
yielded nothing

ls -l /dev/tt*
didnt work

checked this which was dumb
pip list | grep -i tens

went into tt-xla like an idiot
pip list | grep -i tens

now its just checking for a bunch of tenstorrent related stuff cause its stupid as f

i gotta waste another request and intervene.

i wasted the request

now it went into my qwen dir and ran tt-smi
i think it might get stuck looking at this cause its fing stupid.

ok now it wrote 3 test scripts that look like they might be good.

idk what they do tho.
new tabs?


15.4.25
it was using regex insead of the method, changed that
