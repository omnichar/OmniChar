"""Vendored MOSS-Image modeling and pipeline code, taken verbatim from OpenMOSS' release.

    upstream: https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base
    file:     pipeline_moss_image.py
    branch:   main (a Hugging Face model repo; there is no PR)
    commit:   85975793d8b3d432f1b6e27f352abaa4c2a4c0d3
    taken:    2026-11-26

MOSS-Image ships as a diffusers *custom pipeline* loaded with ``trust_remote_code``, which would
fetch and execute code at runtime. Core never does that: the file is reviewed and pinned here and
its classes are constructed directly. It imports only diffusers/transformers/torch, so **no edit
was made at all**; a re-sync is a plain copy.
"""
