#!/bin/bash
jq -r '[.POSITIONS[] | .LABEL]|. as $labels | range(0; length) as $idx | @text "\($idx+1), \($labels[$idx])"'
