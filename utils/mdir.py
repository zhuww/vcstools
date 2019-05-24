#!/usr/bin/env python3

import os
import sys
import logging

logger = logging.getLogger(__name__)


def mdir(path, description, gid=30832):
    """
    Simple function to create directories with the correct group permissions

    The default group ID is 'mwaops' which is 30832 in numerical.
    Here, we try and make sure all directories created by process_vcs
    end up belonging to the user and the group 'mwaops'.
    We also try to give rwx (read, write, execute) permissions and 
    set the sticky bit for both user and group.
    """
    try:
        # TODO: this doesn't carry over permissions correctly to "combined" for some reason...
        os.makedirs(path)
        # we leave the uid unchanged but change gid to mwaops
        os.chown(path, -1, gid)
        os.chmod(path, 0o771)
        os.system("chmod -R g+s {0}".format(path))
    except:
        if (os.path.exists(path)):
            logger.info("{0} Directory Already Exists\n".format(description))
        else:
            logger.error("Could not make new directories\n")
            sys.exit(0)


