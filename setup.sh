# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: setup.sh %
# Date Created:            %date_created: Mon Mar 23 20:15:12 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# One-time setup script for the Vision-Based Sense and Avoid project. Installs 
# all required Python dependencies and the airborne metrics evaluation package.
#
# USAGE INSTRUCTIONS:
#   1. Create and Activate a Virtual Environment.
#   2. Run the Script from Project Root:
#       - bash setup.sh
#
# ------------------------------------------------------------------------------
# -                          L I C E N S I N G                                 -
# ------------------------------------------------------------------------------
# This software and its contents are the intellectual property of Lockheed
# Martin Australia and are protected under applicable copyright and intellectual
# property laws.
#
# All rights reserved. No part of this software may be reproduced, distributed,
# modified, or transmitted in any form or by any means without the prior written
# permission of Lockheed Martin Australia.
#
# Unauthorised use, copying, or distribution of this software is strictly
# prohibited.
# ------------------------------------------------------------------------------

#!/bin/bash
# Install project dependencies.
pip install -r requirements.txt --break-system-packages

# Install the airborne metrics evaluation package.
pip install core/metrics --break-system-packages