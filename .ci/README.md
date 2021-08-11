# This is the Continuous Integration (CI) Area

Making a mausoleum:

1. list all of the desired values to put into the mausoleum in the file: /proj/sdp/data/nexsci.sv.txt
1. create a directory to place working files and the final result: mkdir -p /proj/sdp/${USER}/R_x.y.z  
    where x, y, and z should be numbers like 1.2.3. Any value of x represents some backward compatibily with all other x of the same value. The value y usually indicates what new features have been added. Lastly, z is patching.
1. run inter.sh to build the mausoleum: .ci/inter.sh /proj/sdp/${USER}/R_x.y.z
1. get the file /proj/sdp/${USER}/R_x.y.z to its caretaker
