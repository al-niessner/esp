# Development Tools

First, they can be run anywhere. In other words, you can set your PATH to point to this directory and just type `pp_start.sh` to start a private pipeline (pp) or `pp_stop.sh` to stop a private pipeline. You can also not add it to your PATH and just directly access it with a full or relative like `relative/path/to/excalibur/bash/start.sh` or `/absolute/path/to/excalibur/bash/stop.sh`. The scripts will determine where they are and do the right thing.

Second, they can be run on the mentor cluster or remotely on a laptop. It is not recommended to run post2shelve.sh on a laptop as it could take many days to weeks to complete. The major problem with running on a laptop is access to `/proj/sdp/data`. The simplest way to solve the problem is to do these three items:

1. Add a group like excalibur but it can be any name with the GID 1512 to your laptop then add yourself to the group. This step only be done once in the life of your laptop. May also need to log out or reboot your laptop depending on a bunch of factors that are beyond this readme. Just do as much as you need until the name you chose shows up when you execute `groups` at the command line.
1. May need to copy mentor:/etc/ssl/server.* to your laptop /etc/ssl if they do not already exist. Make sure after copy they have the permissions 600 (RW owner only and root should be owner).
1. Use sshfs to mount /proj/sdp/data
   1. Install sshfs if not on your laptop (once in the life of your laptop)
   1. `sudo mkdir -p /proj/sdp/data` (maybe once in the life of your laptop)
   1. `sudo chown ${USER}:${USER}` (maybe once in the life of your laptop)
   1. run sshfs: `sshfs -o allow_other -o default_permissions -o idmap=user ${USER}@excalibur.jpl.nasa.gov:/proj/sdp/data /proj/sdp/data`
   
When done, be sure to un-mount `/proj/sdp/data` with `fusermount3 -u /proj/sdp/data`.

## Tools

These tools are fairly robust and follow the pythonic guidelines that it should pretty much do out of the box what you want. There may be required values to be set but they will be explicitly noted. Everything else can be left to the default value for the most part. If on your laptop, certainly everything can be left to their defaults.

### post2shelve.sh

Builds a private pipeline database for a private pipeline to use. It requires access to /proj/sdp/data and postgres on the mentor cluster. If using sshfs on a laptop, will also require network access to JPL (VPN).

The person database will end up in `/proj/data/${USER}/db` and named `$USER`.

To run the tool: `post2shelve.sh`

### pp_exec.sh

Execute a development task in the private pipeline context started with `pp_start.sh`. Using exec is faster and more resource friendly than using `pp_worker.sh`.

This script has two required values:
1. `<task name>` is the python package name after excalibur. For instance, if we wanted to run the python packaage `excalibur.cerberus` we would supply `cerberus` as the first argument to the tool.
1. `<target name>` is exactly that. Each star is a target so we would supply the star name as [known by the pipeline](https://excalibur.jpl.nasa.gov:8080/pages/database/targets) like 'GJ 1214'.

and one optional via an environment variable:
1. `RUNID` is an environment variable that can define the run id for the job you going to execute. If not supplied, the value 17 is used.

Putting it all together:

#### use default run id to run excalibur.eclipse for GJ 1214

`pp_exec.sh eclipse "GJ 1214"`

#### use run id 23 to run excalibur.runtime for all targets

`RUNID=23 pp_exec.sh runtime __all__`

In bash, one can define an environment variable for the life of a command by preceeding the command with the variable definition. In most other shells, `RUNID` would have to be defined separately from executing the command.

### pp_reset.sh

The tool will empty the schedule of jobs, optionally archive the database, and then restart the pipeline. It works with both the operational pipeline and private pipelines..

To reset private pipeline with archive: `pp_reset.sh`

To reset private pipeline without archive: `DAWGIE_ARCHIVE=false pp_reset.sh 8080`

To reset operational pipline: `DAWGIE_PIPELINE_HOST=excalibur.jpl.nasa.gov pp_reset.sh 8080`

The script triggers off these environment variables:
- `DAWGIE_FE_PORT` - probably already defined in your bash profile and can be overriden by using the script argument. Default if not defined 8080.
- `DAWGIE_PIPELINE_HOST` - for private pipelines, use computer name running the pipeline like mentor3. Default if not defined excalibur.jpl.nasa.gov.
- `DAWGIE_SSL_PEM_MYSELF` - should default to your certificate and will use the one you have defined in your bash profile. Default if not defined ${HOME}/.ssh/myself.pem.

### pp_start.sh

The tool will start a private pipeline. If you are on the mentor machines, you must supply an argument because TCP/IP ports cannot be shared amoung private pipelines. Nothing will force you to do this except an error saying port already in use. On your laptop, can just use the default port.

To set the port for the pipeline, can either set the environment variable `DAWGIE_FE_PORT`. It is overriden by putting the port number as an argument like `pp_start.sh 12345`. If nothing is defined, then port 9990 is used.

Test:

1. `pp_start.sh` uses what port? [9990]
1. `DAWGIE_FE_PORT=45656 pp_start.sh 12345` uses what port? [12345]
1. `DAWGIE_FE_PORTs=12345 pp_start.sh` uses what port? [9990 - trick question because environment variable name is wrong]
1. `DAWGIE_FE_PORT=54321 pp_start.sh` uses what port? [54321]
1. In .bash_profile `export DAWGIE_FE_PORT=12321`. `pp_start.sh` uses what port? [12321]

On the mentor cluster, it would be best to put `DAWGIE_FE_PORT` in your .bash_profile or your shell's equivalent and set to a value that does not collide with any of the other developers.

Hence, the most common way to start a private pipeline becomes:

`pp_start.sh`

Be aware of these two hidden requirements:
1. DB - the private pipeline database is expected to be in `/proj/sdp/data/$USER/db` and named `${USER}`. If post2shelve.sh was used to create the DB, then this requirement is already met.
1. runtime - the private pipeline settings and knobs are expected to be the file `/proj/sdp/data/runtime/${USER}.xml`.

### pp_stop.sh

Stop the private pipeline started with `pp_start.sh`

### pp_worker.sh

_Under Construction_
