import React from "react";
import { User, Link } from "@nextui-org/react";
import { GithubFetch } from "@Scripts/fordevs"

const contribuitors = [""];
let dataGithub: Array<any> = [];



export default function Contributors() {
    return(
        <User
        name="code1O"
        description={(
            <Link href="" size="sm" isExternal>
                {}
            </Link>
        )}
        avatarProps={{
            src: ""
        }}
        >
        </User>
    )
}