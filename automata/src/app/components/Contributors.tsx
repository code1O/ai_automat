import React from "react";
import { User, Link } from "@nextui-org/react";
import { GithubParse } from "@Scripts/fordevs"

const contribuitors = [""];


export default function Contributors() {
    return(
        <User
        name="code1O"
        description={(
            <Link href="" size="sm" isExternal>
                @code1O
            </Link>
        )}
        avatarProps={{
            src: ""
        }}
        >
        </User>
    )
}