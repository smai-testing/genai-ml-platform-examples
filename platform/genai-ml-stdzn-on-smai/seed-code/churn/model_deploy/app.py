# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: MIT-0
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from aws_cdk import App, Environment
from deploy_endpoint.deploy_endpoint_stack import DeployEndpointStack
from config.constants import (
    DEPLOY_ACCOUNT,
    DEFAULT_DEPLOYMENT_REGION,
    MODEL_PACKAGE_GROUP_NAME,
)


if not MODEL_PACKAGE_GROUP_NAME:
    raise ValueError(
        "model_package_group_name is empty in config/deploy_config.json. "
        "The project template fills this in when it seeds the model folder."
    )

app = App()

dev_env = Environment(
    account=DEPLOY_ACCOUNT,
    region=DEFAULT_DEPLOYMENT_REGION
)

# One stack per model. Both mono-repos hold many models, and two models
# deploying into the same account must not share a stack name or the second
# one would tear down the first one's endpoint.
endpoint_stack = DeployEndpointStack(
    app,
    f"sagemaker-endpoint-{MODEL_PACKAGE_GROUP_NAME}",
    env=dev_env
)

app.synth()