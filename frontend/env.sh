#!/bin/sh

# Recreate config file
rm -rf ./env-config.js
touch ./env-config.js

# Add assignment 
echo "window._env_ = {" >> ./env-config.js

# Check if .env file exists, if not use environment variables only
if [ -f ".env" ]; then
  # Read each line in .env file
  # Each line represents key=value pairs
  while read -r line || [ -n "$line" ];
  do
    # Split env variables by character `=`
    if printf '%s\n' "$line" | grep -q -e '='; then
      varname=$(printf '%s\n' "$line" | sed -e 's/=.*//')
      varvalue=$(printf '%s\n' "$line" | sed -e 's/^[^=]*=//')
    fi

    # Read value of current variable if exists as Environment variable
    eval value=\$$varname
    # Otherwise use value from .env file
    [ -z "$value" ] && value=${varvalue}
    
    # Append configuration property to JS file
    echo "  $varname: \"$value\"," >> ./env-config.js
  done < .env
else
  # Use environment variables passed from docker-compose
  # Add all REACT_APP_ prefixed environment variables
  env | grep "^REACT_APP_" | while read -r line; do
    varname=$(echo "$line" | cut -d= -f1)
    varvalue=$(echo "$line" | cut -d= -f2-)
    echo "  $varname: \"$varvalue\"," >> ./env-config.js
  done
fi

echo "}" >> ./env-config.js
