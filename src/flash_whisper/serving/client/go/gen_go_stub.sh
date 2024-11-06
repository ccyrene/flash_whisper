PACKAGE="grpc-client"

# Clean up before
rm -rf ./core

# Get the proto files from the common repo
mkdir core && cp common/protobuf/*.proto core/.

for i in ./core/*.proto
do 
  # https://developers.google.com/protocol-buffers/docs/reference/go-generated#package
  echo "option go_package = \"./${PACKAGE}\";" >> $i
done

# Requires protoc and protoc-gen-go plugin: https://grpc.io/docs/protoc-installation/
protoc -I core --go-grpc_out="./" --go_out="./" core/*.proto