group "default" {
  targets = ["client"]
}

target "client" {
  context    = "./client"

  # Specify output type as a local directory
  output = [
    "type=local,dest=./client/dist"
  ]
}