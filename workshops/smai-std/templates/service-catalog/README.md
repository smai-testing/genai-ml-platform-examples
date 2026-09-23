# Service Catalog Product Templates

These CloudFormation templates are designed to be deployed as **AWS Service Catalog products**. They decompose the monolithic `sagemaker-domain-with-vpc.yaml` into three modular, independently provisionable components.

## Products

| Product | Template | Description | Dependencies |
|---------|----------|-------------|-------------|
| **1. Networking** | `sc-product-networking.yaml` | VPC, subnets, NAT, VPC Endpoints, security groups | None |
| **2. IAM Roles** | `sc-product-iam-roles.yaml` | SageMaker execution roles and managed policies | Product 1 (VPC ID) |
| **3. SageMaker Domain** | `sc-product-sagemaker-domain.yaml` | Domain, user profiles, MLflow, S3, Access Grants, Glue | Products 1 & 2 |

## Provisioning Order

Products must be provisioned in order (1 → 2 → 3) because each product's outputs feed as inputs to subsequent products.

## Usage

### As Service Catalog Products (Recommended)

1. Upload templates to an S3 bucket
2. Create a Service Catalog portfolio
3. Add each template as a product in the portfolio
4. Add launch constraints with an IAM role
5. Grant access to end users

See **Lab 1C** instructions for detailed walkthrough.

### Standalone CloudFormation Stacks

You can also deploy these as regular CloudFormation stacks if Service Catalog is not required:

```bash
# Deploy networking
aws cloudformation create-stack --stack-name ml-networking \
  --template-body file://sc-product-networking.yaml \
  --parameters ParameterKey=ProjectName,ParameterValue=bank-marketing-prediction

# Deploy IAM roles (after networking completes)
aws cloudformation create-stack --stack-name ml-roles \
  --template-body file://sc-product-iam-roles.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters ParameterKey=ProjectName,ParameterValue=bank-marketing-prediction \
    ParameterKey=VPCId,ParameterValue=<vpc-id-from-stack-1>

# Deploy SageMaker domain (after roles completes)
aws cloudformation create-stack --stack-name ml-domain \
  --template-body file://sc-product-sagemaker-domain.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters ParameterKey=ProjectName,ParameterValue=bank-marketing-prediction \
    ParameterKey=VPCId,ParameterValue=<vpc-id> \
    ParameterKey=PrivateSubnetId,ParameterValue=<subnet-1> \
    ParameterKey=PrivateSubnet2Id,ParameterValue=<subnet-2> \
    ParameterKey=SecurityGroupId,ParameterValue=<sg-id> \
    ParameterKey=UserARoleArn,ParameterValue=<role-arn-a> \
    ParameterKey=UserBRoleArn,ParameterValue=<role-arn-b>
```
