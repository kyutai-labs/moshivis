# moshi-client

Frontend for the demo.

## Quickstart

To start developping, you will need a basic environment with NodeJS, for instance:
```bash
cd client
micromamba create -n node22 python=3.10
micromamba activate node22
micromamba install nodejs=22.11
# install
npm install
```
Alternatively, you can use [NVM](https://github.com/nvm-sh/nvm) to help you manage your node version and make sure you're on the recommended version for this project. If you do so run, `nvm use`.

To run the client in dev mode, use:
```bash
# typically will start on port 5173
npm run dev
```

When you're satisfied, build the client (in `dist` directory) that will be used as
static dir by the  different  backends:
```bash
npm run build
```

If Docker is available, you can skip all the previous steps and just run

```
docker buildx bake
```
from the root of this repository. It will output the static sources for the website in `client/dist`.

### License

The present code is provided under the MIT license.
