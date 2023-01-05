runpy () {
    docker run \
        -it \
        --rm \
        --init \
        --gpus '"device=0"' \
        --shm-size 16G \
        --volume="$HOME/.cache/torch:/root/.cache/torch" \
        --volume="$PWD:/workspace" \
        session-aware-bert4rec \
        python "$@"
}

# runpy entry.py onmo/bert4rec/vanilla
runpy entry.py onmo/sasrec