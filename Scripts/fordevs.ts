/**
 *  This module contains useful tools for developers.
 * 
 * This tools are made for: app development, app decoration, & other tasks
 *  
 */

require('dotenv').config();

import { BaseResponse, getJson } from "serpapi";

export class GithubFetch {
    public user: string;
    constructor(user: string) {
        this.user = user;
    }
    private async fetchData() {
        const api_url = `https://api.github.com/users/${this.user}`;
        const response = await fetch(api_url, {method: "GET"});
        return response.json();
    }
    private async fetchMisc() {
        return await this.fetchData().then(user => {
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
    public async Misscelaneous() {
        const data = await this.fetchMisc();
        return data;
    }
}

export class GoogleSearch {
    query: string;
    private api = process.env.google_key;
    constructor(query: string) {
        this.query = query;
    }
    private async fetcher() {
        const result = await getJson({
            q: this.query,
            hl: "en",
            gl: "us",
            google_domain: "google.com",
            api_key: this.api
        }, (json: BaseResponse) => {
            return json
        })
        return result
    }
    public async fetch(){
        return await this.fetcher().then(data => {
            return data
        })
    }
}