# Fix MongoDB Security Group - Quick Guide

## Issue
The RAG application is configured to use the remote MongoDB, but the connection is failing because the MongoDB EC2 security group is blocking port 27017.

## Quick Fix - AWS Console

1. **Go to AWS EC2 Console**
   - Navigate to: https://console.aws.amazon.com/ec2/
   - Make sure you're in the **ap-southeast-2** region (Asia Pacific Sydney)

2. **Find the MongoDB Instance**
   - In EC2 → Instances
   - Search for: `3-25-73-98` or look for instance with hostname containing `ec2-3-25-73-98`

3. **Update Security Group**
   - Click on the MongoDB instance
   - Go to **Security** tab
   - Click on the **Security Group** link
   - Click **Edit inbound rules**
   - Click **Add rule**
   - Configure:
     - **Type**: Custom TCP
     - **Port range**: 27017
     - **Source**: Custom → Enter `98.93.155.138/32` (RAG instance IP)
     - **Description**: Allow RAG instance MongoDB access
   - Click **Save rules**

4. **Wait 10-15 seconds** for the security group to propagate

5. **Test Connection**
   The RAG application will automatically retry the connection. Check with:
   ```bash
   ssh -i key/RAGServerKey.pem ec2-user@ec2-98-93-155-138.compute-1.amazonaws.com
   docker logs rag_api | tail -20
   curl http://localhost/health
   ```

## Alternative: Allow from Anywhere (Less Secure - for Testing Only)

If you need quick access for testing:

1. In the security group, add rule:
   - **Type**: Custom TCP
   - **Port range**: 27017
   - **Source**: 0.0.0.0/0
   - **Description**: Temporary - allow MongoDB access

2. **Important**: This opens MongoDB to the entire internet. Only use for testing!

## Verify Fix

After updating the security group, the RAG application should automatically connect. You can verify:

```bash
# From RAG instance
curl http://localhost/health

# Should show:
# {
#     "status": "healthy",
#     "embedding_model": "loaded",
#     "reranker_model": "loaded",
#     "database": "connected"  <-- This should now be "connected"
# }
```

## Current Configuration

- **RAG Instance**: ec2-98-93-155-138.compute-1.amazonaws.com (IP: 98.93.155.138)
- **MongoDB Instance**: ec2-3-25-73-98.ap-southeast-2.compute.amazonaws.com
- **Database**: medicaldiagnosissystem
- **Port**: 27017

The RAG application is already configured correctly - it just needs the security group to allow the connection.

