/**
 *  This module contains useful tools for developers.
 * 
 * This tools are made for: app development, app decoration, & other tasks
 *  
 */

import { PythonShell } from "python-shell";

export class GithubParse {
    protected user: string;
    constructor(username: string) {
        this.user = username;
    }
    private async fetchData() {
        const api_url = `https://api.github.com/users/${this.user}`;
        const response = await fetch(api_url, {method: "GET"});
        return response.json();
    }
    public Misscelaneous() {
        return this.fetchData().then(user => {
            const objects = {
                avatar: user.avatar_url,
                twitter: user.twitter_username,
                bio: user.bio,
                name: user.name,
                followers: user.followers,
                following: user.following,
                hireable: user.hireable
            }
            return objects;
        })
    }
}


export function runPython(){}