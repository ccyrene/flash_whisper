FROM golang:1.23

WORKDIR /workspace

COPY ./client/go/go.mod ./client/go/go.sum ./
RUN go mod download && go mod verify

COPY ./client/go/*.go .
COPY ./client/go/grpc-client grpc-client
RUN go build -v -o .

ENTRYPOINT ["./goclient"]