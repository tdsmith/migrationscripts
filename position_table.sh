#!/bin/bash
jq -c '[.POSITIONS[] | .LABEL]|. as $labels | range(0; length) as $idx | [$idx+1, $labels[$idx]]'
