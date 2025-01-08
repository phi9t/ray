# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


from spack.package import *


class PyRay(PythonPackage):
    """Ray provides a simple, universal API for building distributed applications."""

    homepage = "https://github.com/ray-project/ray"
    target_version = "2.40.0"
    url = f"https://github.com/ray-project/ray/archive/ray-{target_version}.tar.gz"

    license("Apache-2.0")

    version(
        target_version,
        sha256="9eec094e79b34fad48b736205752e7e57b4afff1153780314c91bd8ef5a373fe",
    )

    depends_on("bazel@6.5.0", type="build")
    depends_on("npm", type="build")
    depends_on("py-setuptools", type="build")

    depends_on("python@3.6:3.10", type=("build", "run"))

    # grpc and protobuf need to be fixed to compatible versions.
    depends_on("py-grpcio@1.60.1", type=("build", "run"))
    depends_on("py-protobuf", type=("build", "run"))

    depends_on("py-click@7:8.0.4", type=("build", "run"))
    depends_on("py-cython@0.29.37:", type="build")
    depends_on("py-filelock", type=("build", "run"))
    depends_on("py-frozenlist", type=("build", "run"))
    depends_on("py-jsonschema", type=("build", "run"))
    depends_on("py-msgpack@1.0.0:2.0.0", type=("build", "run"))
    depends_on("py-numpy@1.24.4:", type=("build", "run"))
    depends_on("py-pyarrow@6.0.1:")
    depends_on("py-pyyaml", type=("build", "run"))
    depends_on("py-requests", type=("build", "run"))
    depends_on("py-virtualenv", when="@2.0.1", type=("build", "run"))

    build_directory = "python"

    def patch(self):
        filter_file(
            'bazel_flags = ["--verbose_failures"]',
            f'bazel_flags = ["--verbose_failures", "--jobs={make_jobs}"]',
            join_path("python", "setup.py"),
            string=True,
        )

    def setup_build_environment(self, env):
        env.set("SKIP_THIRDPARTY_INSTALL", "1")

    # Compile the dashboard npm modules included in the project
    @run_before("install")
    def build_dashboard(self):
        with working_dir(join_path("python", "ray", "dashboard", "client")):
            npm = which("npm")
            npm("ci")
            npm("run", "build")
