##
# Project Title
#
# @file
# @version 0.1

SOURCE_SUFFIX = dockerfile
TARGET_SUFFIX = container_image_tag
IMAGE_REPO = sygaldry/ray_builder

all: base.$(TARGET_SUFFIX)

base.$(TARGET_SUFFIX): base.$(SOURCE_SUFFIX)
	docker build -f $< -t $(IMAGE_REPO):base . && touch $@


# end
